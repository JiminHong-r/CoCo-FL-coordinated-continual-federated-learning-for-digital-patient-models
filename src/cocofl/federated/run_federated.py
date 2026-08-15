"""WP4 main script: federated FFN-adapter training across hospital silos.

Loads a WP2 (or WP2+notes) checkpoint, splits patients in the test directory
into ``num_hospitals`` via Dirichlet, runs ``rounds`` rounds of FedAvg on the
adapter (+ optional head), and tracks AUC per round.

Per-round evaluation:
  * per-hospital local AUC (each hospital evaluates the global model on its
    own held-out fraction)
  * global pooled AUC on a separate population test split

Outputs a CSV ``federated_history.csv`` to ``output_dir``.

Usage::

    cocofl_federated \\
        model_path=/path/to/best_model.pt \\
        train_dir=/path/to/mimic-train \\
        input_dir=/path/to/mimic-test \\
        output_dir=federated_artifacts \\
        +num_hospitals=3 +alpha=1.0 +rounds=10 +local_epochs=1 +bottleneck=64
"""
from __future__ import annotations

import copy
import pickle
import random
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from ..constants import SpecialToken as ST
from ..constants import STATIC_DATA_FN
from ..data import HospitalMortalityDataset, ICUAdmissionDataset, ReadmissionDataset
from ..data.static_features import (
    CATEGORICAL_CARDINALITIES, NUMERICAL_DIM, TFNTimelineDataset,
    build_static_arrays, collate_with_static,
)
from ..models.classifier import TFNDownstreamClassifier
from ..models.backbone import TFNFullModel
from ..adapters import attach_ffn_adapter
from ..personalisation.evaluate import eval_auc
from ..personalisation.run_ablation import (
    POSITIVE_TOKENS, TASKS, make_collate, make_label_fn,
)
from ..vocabulary import Vocabulary
from .fedavg import (
    FedRoundConfig, initialize_server_state, load_federated_state, run_round,
)
from .hospital_split import dirichlet_split_patients, summarize_split


def build_population_classifier(model_path: Path, device: torch.device) -> TFNDownstreamClassifier:
    ck = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ck["model_config"]
    backbone = TFNFullModel(
        vocab_size=cfg.vocab_size, n_embd=cfg.n_embd, n_layer=cfg.n_layer,
        n_positions=cfg.n_positions,
        categorical_cardinalities=tuple(cfg.categorical_cardinalities),
        numerical_dim=cfg.numerical_dim, use_static=cfg.use_static,
        num_heads=getattr(cfg, "num_heads", 2),
    )
    backbone.load_state_dict(ck["model"])
    return TFNDownstreamClassifier(backbone, num_classes=1).to(device)


def build_hospital_loaders(
    dataset, split, collate_fn, batch_size: int = 1,
    held_out_frac: float = 0.2, seed: int = 42,
) -> tuple[dict[int, DataLoader], dict[int, DataLoader]]:
    """Returns (train_loaders, eval_loaders) per hospital.

    Each hospital's indices are shuffled and split into a train/eval portion.
    Held-out fraction is fixed across hospitals for fair comparison.
    """
    train_loaders, eval_loaders = {}, {}
    rng = random.Random(seed)
    for hid, indices in split.indices_per_hospital.items():
        if not indices:
            continue
        idxs = list(indices)
        rng.shuffle(idxs)
        n_eval = max(1, int(len(idxs) * held_out_frac))
        eval_idx = idxs[:n_eval]
        train_idx = idxs[n_eval:]
        if not train_idx:
            train_idx = idxs  # tiny hospital — fall back to using all for both
        train_loaders[hid] = DataLoader(
            Subset(dataset, train_idx), batch_size=batch_size,
            shuffle=True, collate_fn=collate_fn,
        )
        eval_loaders[hid] = DataLoader(
            Subset(dataset, eval_idx), batch_size=batch_size,
            shuffle=False, collate_fn=collate_fn,
        )
    return train_loaders, eval_loaders


def evaluate_global(template, server_state, dataset, indices, collate_fn,
                     label_fn, device, batch_size: int = 1):
    clf = copy.deepcopy(template)
    load_federated_state(clf, server_state)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_fn)
    return eval_auc(clf, loader, label_fn, device)


def evaluate_per_hospital(template, server_state, eval_loaders, label_fn, device):
    clf = copy.deepcopy(template)
    load_federated_state(clf, server_state)
    per_hospital: dict[int, float | None] = {}
    for hid, loader in eval_loaders.items():
        res = eval_auc(clf, loader, label_fn, device)
        per_hospital[hid] = res.auc
    return per_hospital


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = Path(cfg.train_dir)
    test_dir = Path(cfg.input_dir)
    model_path = Path(cfg.model_path)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_hospitals = int(getattr(cfg, "num_hospitals", 3))
    alpha = float(getattr(cfg, "alpha", 1.0))
    rounds = int(getattr(cfg, "rounds", 10))
    lm_rounds = int(getattr(cfg, "lm_rounds", 5))  # for two_stage / lm
    local_epochs = int(getattr(cfg, "local_epochs", 1))
    local_steps = getattr(cfg, "local_steps_per_epoch", None)
    bottleneck = int(getattr(cfg, "bottleneck", 64))
    lr = float(getattr(cfg, "lr", 1e-3))
    lm_lr = float(getattr(cfg, "lm_lr", 3e-4))  # TFN LM lr default
    weight_decay = float(getattr(cfg, "weight_decay", 0.0))
    include_head = bool(getattr(cfg, "include_head_in_fedavg", True))
    n_pop_eval = int(getattr(cfg, "n_pop_eval", 2000))
    seed = int(getattr(cfg, "seed", 42))
    n_positions = int(getattr(cfg, "n_positions", 2048))
    # phase: "downstream" (default, current behaviour), "lm", or "two_stage"
    phase = str(getattr(cfg, "phase", "downstream"))
    assert phase in ("downstream", "lm", "two_stage"), f"unknown phase: {phase!r}"
    # When ``splits_dir`` is set we load fixed federated splits from JSON.
    splits_dir = getattr(cfg, "splits_dir", None)


    def _build_hospital_split(dataset):
        """Returns a HospitalSplit bound to ``dataset`` for the configured α."""
        if splits_dir is not None:
            from .fixed_splits import load_split, as_hospital_split
            split_data = load_split(splits_dir, alpha)
            train_split = as_hospital_split(split_data, partition="train")
            train_split.bind_to_dataset(dataset)
            return train_split
        all_pids = list(test_static.keys())
        s = dirichlet_split_patients(all_pids, num_hospitals, alpha, seed=seed)
        s.bind_to_dataset(dataset)
        return s

    Vocabulary.from_path(test_dir)

    with open(test_dir / STATIC_DATA_FN, "rb") as f:
        test_static = pickle.load(f)
    test_cat, test_num = build_static_arrays(test_static)
    collate_fn = make_collate(test_cat, test_num)

    # Reuse population classifier across tasks (the backbone doesn't change)
    pop_clf = build_population_classifier(model_path, device)
    print(f"Loaded population classifier from {model_path}")

    all_rows = []

    # --- Shared LM phase (only if phase in {"lm", "two_stage"}) ---
    # The LM phase trains a single global adapter on next-token CE using the
    # tokenised TFNTimelineDataset; the resulting state is then handed to the
    # downstream phase (one adapter per task) when phase == "two_stage".
    lm_server_state = None
    lm_template = None
    if phase in ("lm", "two_stage"):
        print(f"\n=== LM federated phase (hospitals={num_hospitals}, alpha={alpha}, "
              f"rounds={lm_rounds}) ===")
        # Use the TEST dir's tokenized timeline (consistent with user's "test patients split" choice)
        lm_ds = TFNTimelineDataset(str(test_dir), n_positions=n_positions)
        split = _build_hospital_split(lm_ds)
        print(summarize_split(split))

        lm_train_loaders, _ = build_hospital_loaders(
            lm_ds, split, collate_with_static, batch_size=1, held_out_frac=0.2, seed=seed,
        )

        lm_template = copy.deepcopy(pop_clf)
        lm_server_state = initialize_server_state(lm_template, bottleneck=bottleneck,
                                                    include_head=False)  # head irrelevant for LM
        lm_round_cfg = FedRoundConfig(
            local_epochs=local_epochs, local_steps_per_epoch=local_steps,
            lr=lm_lr, weight_decay=weight_decay,
            include_head_in_fedavg=False,
        )
        for r in range(1, lm_rounds + 1):
            lm_server_state, counts = run_round(
                lm_server_state, lm_template, lm_train_loaders, lm_round_cfg, device,
                objective="lm",
            )
            print(f"  LM round {r}: counts={counts}")
            all_rows.append({
                "task": "_LM_", "round": r, "phase": "lm_fedavg",
                **{f"hosp_{h}_count": c for h, c in counts.items()},
            })
            pd.DataFrame(all_rows).to_csv(output_dir / "federated_history.csv", index=False)

    # --- Downstream phase per task (skipped if phase == "lm") ---
    if phase == "lm":
        print(f"\nphase='lm' selected — skipping downstream tasks.")
        print(f"Wrote {output_dir / 'federated_history.csv'}")
        return

    for task_name, dataset_cls in TASKS.items():
        print(f"\n=== Task: {task_name} (hospitals={num_hospitals}, alpha={alpha}, rounds={rounds}) ===")
        test_ds = dataset_cls(test_dir)
        label_fn = make_label_fn(task_name)

        split = _build_hospital_split(test_ds)
        print(summarize_split(split))

        train_loaders, eval_loaders = build_hospital_loaders(
            test_ds, split, collate_fn, batch_size=1, held_out_frac=0.2, seed=seed,
        )
        pop_eval_indices = sorted({i for ld in eval_loaders.values()
                                    for i in ld.dataset.indices})[:n_pop_eval]

        template = copy.deepcopy(pop_clf)
        server_state = initialize_server_state(template, bottleneck=bottleneck,
                                                include_head=include_head)
        # If we ran the LM phase, seed the downstream adapter with the LM state.
        # head was not part of the LM state, so it stays at its fresh init.
        if lm_server_state is not None:
            adapter_only = {k: v for k, v in lm_server_state.items()
                            if k.startswith("ffn_adapter.")}
            load_federated_state(template, adapter_only)
            server_state = {**server_state, **adapter_only}
            print("  seeded adapter from LM phase")
        print(f"Server adapter initialised: {sum(v.numel() for v in server_state.values()):,} params transmitted/round")

        round_cfg = FedRoundConfig(
            local_epochs=local_epochs, local_steps_per_epoch=local_steps,
            lr=lr, weight_decay=weight_decay,
            include_head_in_fedavg=include_head,
        )

        baseline_global = evaluate_global(template, server_state, test_ds,
                                            pop_eval_indices, collate_fn, label_fn, device)
        baseline_per_hosp = evaluate_per_hospital(template, server_state, eval_loaders,
                                                    label_fn, device)
        print(f"  round 0: global AUC = {baseline_global.auc}; per-hospital = {baseline_per_hosp}")
        all_rows.append({
            "task": task_name, "round": 0, "phase": "baseline",
            "global_auc": baseline_global.auc,
            **{f"hosp_{h}_auc": v for h, v in baseline_per_hosp.items()},
        })

        for r in range(1, rounds + 1):
            server_state, counts = run_round(
                server_state, template, train_loaders, round_cfg, device,
                objective="downstream", label_fn=label_fn,
            )
            global_res = evaluate_global(template, server_state, test_ds,
                                          pop_eval_indices, collate_fn, label_fn, device)
            per_hosp = evaluate_per_hospital(template, server_state, eval_loaders,
                                              label_fn, device)
            print(f"  round {r}: global AUC = {global_res.auc}; "
                  f"per-hospital = {per_hosp}; counts = {counts}")
            all_rows.append({
                "task": task_name, "round": r, "phase": "fedavg",
                "global_auc": global_res.auc,
                **{f"hosp_{h}_auc": v for h, v in per_hosp.items()},
                **{f"hosp_{h}_count": c for h, c in counts.items()},
            })
            pd.DataFrame(all_rows).to_csv(output_dir / "federated_history.csv", index=False)

    print(f"\nWrote {output_dir / 'federated_history.csv'}")
    print(pd.DataFrame(all_rows).to_string(index=False))


if __name__ == "__main__":
    main()
