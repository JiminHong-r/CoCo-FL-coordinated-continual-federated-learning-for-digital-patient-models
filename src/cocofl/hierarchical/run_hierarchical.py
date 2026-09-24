"""WP4 main: hierarchical bidirectional adaptation (CoCo-FL).

Population (frozen)  -+-> Cohort A adapter <-- aggregated patient deltas
                       |
                       +-> Cohort B adapter <-- ...
                       |
                       +-> Cohort C adapter <-- ...
                            ^   |
              EMA blend     |   v (init from cohort)
                            +-- Patient adapters

Usage::

    cocofl_hierarchical \\
        model_path=$PROJECT/models/population/best_model.pt \\
        train_dir=$PROJECT/data/tokenized/mimic_train \\
        input_dir=$PROJECT/data/tokenized/mimic_test \\
        output_dir=$PROJECT/federated_artifacts \\
        +num_hospitals=3 +alpha=1.0 +rounds=5 \\
        +patients_per_round=5 +n_patient_iters=50
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

from ..constants import STATIC_DATA_FN
from ..data.static_features import build_static_arrays
from ..models.classifier import TFNDownstreamClassifier
from ..models.backbone import TFNFullModel
from ..adapters import attach_ffn_adapter, load_level_state_dict
from ..evaluation.batching import make_eval_loader, make_train_loader
from ..personalisation.evaluate import eval_auc
from ..personalisation.loop import per_patient_subsets, split_indices_by_time
from ..personalisation.run_ablation import (
    TASKS, make_collate, make_label_fn,
)
from ..vocabulary import Vocabulary
from ..federated.hospital_split import dirichlet_split_patients, summarize_split
from .coordinator import CoordinatorConfig, HierarchicalCoordinator
from .cohort import CohortConfig
from .patient import PatientConfig


def build_population_classifier(model_path: Path, device: torch.device,
                                 head_path: Path | None = None) -> TFNDownstreamClassifier:
    """Population model = pretrained backbone + (optionally) its trained head.

    The checkpoint holds only the backbone and its language-model head, so
    without ``head_path`` the downstream classifier head is RANDOMLY
    INITIALISED. Every WP4/WP5 method then starts from backbone + noise:
    ``population_only`` scores at chance, and Δ_cohort measures "was a head
    trained at all" rather than whether cohort adaptation helps. Pass the head
    fitted on the 40%% reserve (``cocofl_finetune +save_head_dir=...``) so the
    population level is the WP3 population model the spec assumes.
    """
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
    clf = TFNDownstreamClassifier(backbone, num_classes=1).to(device)
    if head_path is not None:
        hp = Path(head_path)
        if not hp.is_file():
            raise FileNotFoundError(
                f"population head not found at {hp}. Fit it with "
                f"`cocofl_finetune +probe=balanced +save_head_dir=<dir>`; running "
                f"without it leaves a randomly initialised classifier head and "
                f"makes population_only a chance-level baseline."
            )
        blob = torch.load(hp, map_location=device, weights_only=False)
        clf.classifier.load_state_dict(blob["classifier"])
        print(f"  population head: {hp.name} "
              f"(reserve-fit, test AUROC {blob.get('test_auroc')})")
    return clf


def build_hospital_train_loader(dataset, indices, collate_fn, batch_size=1, seed=42):
    """Shuffled cohort-training loader for one centre's partition.

    Delegates to ``make_train_loader`` so that ``batch_size > 1`` is actually
    usable: downstream windows are variable-length, so a plain DataLoader would
    fail in ``collate``'s ``torch.stack`` above batch 1. Length-bucketing keeps
    each batch homogeneous, so no padding enters the T-LSTM recurrence.
    """
    return make_train_loader(dataset, indices, collate_fn,
                             batch_size=batch_size, seed=seed)


def build_patient_streams(dataset, pids, collate_fn, train_frac=0.5,
                           min_samples=60, max_patients=20, eval_batch_size=1,
                           patient_batch_size=8):
    """Returns a list of ``(patient_id, fit_loader, val_loader, eval_loader)``.

    ``min_samples`` is the statistically load-bearing knob. At the default of 4
    a qualifying patient contributes 2 training windows and — because the split
    is 50/50 by time — only **2 evaluation windows**, so their personal AUROC
    can only take the values {0, 0.5, 1}. Raising it (e.g. 20 → 10 eval windows)
    is what makes per-patient AUROC, and therefore Δ_patient, a measurable
    quantity rather than a coin flip. It costs patients: the mean window count
    per admitted patient is ~2.4, so a higher threshold selects a smaller and
    more admission-heavy subset.
    """
    buckets = per_patient_subsets(dataset, min_samples=min_samples)
    streams = []
    for pid in pids:
        if pid not in buckets:
            continue
        early, late = split_indices_by_time(dataset, buckets[pid], train_frac=train_frac)
        if not early or not late:
            continue
        # Three-way, chronological: fit / val / eval. The val slice is what lets
        # the patient adapter be ACCEPTED OR REJECTED on held-out data — without
        # it the only held-out windows are the eval ones, and selecting on those
        # would be selecting on the test set. The cohort layer has had this
        # protection since model selection was added; the patient layer did not,
        # which is why delta_patient averaged -0.099 while delta_cohort sat at 0.
        fit, val = split_indices_by_time(dataset, early, train_frac=0.7)
        if not fit or not val:
            continue
        # Length-bucketed batches, like every other training loader. At batch 1
        # with n_iters=10 the patient adapter saw TEN windows to fit 66 K
        # parameters — it memorised them and then failed validation, which is
        # why patient selection rejected every adapter. Same defect the WP3
        # probe had (batch_size=1 on a starved step budget); it was never
        # checked at the patient level.
        t = make_train_loader(dataset, fit, collate_fn,
                              batch_size=patient_batch_size, seed=42)
        v = make_eval_loader(dataset, val, collate_fn, batch_size=eval_batch_size)
        e = make_eval_loader(dataset, late, collate_fn, batch_size=eval_batch_size)
        # The patient id travels WITH the loaders. Spec WP5 axis 2 asks for
        # "analysis of when and for which patient subgroups personalisation is
        # most beneficial", which is impossible from pooled per-hospital numbers:
        # it needs a per-patient record joinable to that patient's attributes.
        # Discarding pid here made that analysis unreconstructable after the fact.
        streams.append((pid, t, v, e))
        if len(streams) >= max_patients:
            break
    return streams


def evaluate_with_states(template, cohort_state, patient_state, eval_loader,
                          label_fn, device, cohort_bot, patient_bot):
    """Build a classifier with given cohort + patient adapter states and AUC-eval."""
    clf = copy.deepcopy(template)
    attach_ffn_adapter(clf.backbone, bottleneck=cohort_bot, attr="cohort_adapter")
    load_level_state_dict(clf, cohort_state, attr="cohort_adapter")
    if patient_state is not None:
        attach_ffn_adapter(clf.backbone, bottleneck=patient_bot, attr="patient_adapter")
        load_level_state_dict(clf, patient_state, attr="patient_adapter")
    return eval_auc(clf, eval_loader, label_fn, device)


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
    rounds = int(getattr(cfg, "rounds", 5))
    patients_per_round = int(getattr(cfg, "patients_per_round", 5))
    max_patients_per_hospital = int(getattr(cfg, "max_patients_per_hospital", 20))
    n_patient_iters = int(getattr(cfg, "n_patient_iters", 50))
    cohort_bottleneck = int(getattr(cfg, "cohort_bottleneck", 64))
    # patient_bottleneck must match cohort_bottleneck for upward propagation
    # (the deltas have the same shape as the cohort adapter weights).
    patient_bottleneck = int(getattr(cfg, "patient_bottleneck", cohort_bottleneck))
    cohort_local_steps = getattr(cfg, "cohort_local_steps", 100)
    cohort_local_epochs = int(getattr(cfg, "cohort_local_epochs", 1))
    cohort_lr = float(getattr(cfg, "cohort_lr", 1e-3))
    patient_lr = float(getattr(cfg, "patient_lr", 1e-3))
    eta_cohort = float(getattr(cfg, "eta_cohort", 0.3))
    eta_population = float(getattr(cfg, "eta_population", 0.1))
    population_update_freq = int(getattr(cfg, "population_update_freq", 3))
    n_pop_eval = int(getattr(cfg, "n_pop_eval", 500))
    seed = int(getattr(cfg, "seed", 42))
    tasks_filter = getattr(cfg, "tasks", None)
    # When ``splits_dir`` is set we load fixed federated splits (train/val/test
    # per hospital) from JSON instead of redrawing the Dirichlet inline.
    splits_dir = getattr(cfg, "splits_dir", None)
    eval_per_hosp_cfg = int(getattr(cfg, "eval_per_hospital", 200))

    Vocabulary.from_path(test_dir)
    with open(test_dir / STATIC_DATA_FN, "rb") as f:
        test_static = pickle.load(f)
    test_cat, test_num = build_static_arrays(test_static)
    collate_fn = make_collate(test_cat, test_num)

    population_clf = build_population_classifier(model_path, device)
    print(f"Loaded population classifier from {model_path}")

    all_rows = []
    selected_tasks = {k: v for k, v in TASKS.items()}
    if tasks_filter is not None:
        if isinstance(tasks_filter, str):
            keep = set(s.strip() for s in tasks_filter.split(","))
        else:
            keep = set(str(s).strip() for s in tasks_filter)
        selected_tasks = {k: v for k, v in TASKS.items() if k in keep}
    print(f"running tasks: {list(selected_tasks.keys())}")
    for task_name, dataset_cls in selected_tasks.items():
        print(f"\n=== Task: {task_name} (hospitals={num_hospitals}, alpha={alpha}, "
              f"rounds={rounds}, patients/round/hospital={patients_per_round}) ===")
        test_ds = dataset_cls(test_dir)
        label_fn = make_label_fn(task_name)

        all_pids = list(test_static.keys())
        hospital_train_loaders = {}
        hospital_eval_loaders = {}
        hospital_patient_streams = {}

        if splits_dir is not None:
            # Fixed split path: load train + test partitions from disk.
            from ..federated import load_split, as_hospital_split
            split_data = load_split(splits_dir, alpha)
            train_split = as_hospital_split(split_data, partition="train")
            test_split = as_hospital_split(split_data, partition="test")
            train_split.bind_to_dataset(test_ds)
            test_split.bind_to_dataset(test_ds)
            print(summarize_split(train_split))
            for hid, train_idx in train_split.indices_per_hospital.items():
                if not train_idx:
                    continue
                hospital_train_loaders[hid] = build_hospital_train_loader(
                    test_ds, train_idx, collate_fn,
                )
                hospital_patient_streams[hid] = build_patient_streams(
                    test_ds, train_split.pids_per_hospital[hid], collate_fn,
                    max_patients=max_patients_per_hospital,
                )
                print(f"  hosp {hid}: train={len(train_idx)}, patient_streams="
                      f"{len(hospital_patient_streams[hid])}")
            for hid, test_idx in test_split.indices_per_hospital.items():
                if not test_idx:
                    continue
                hospital_eval_loaders[hid] = DataLoader(
                    Subset(test_ds, test_idx[:eval_per_hosp_cfg]),
                    batch_size=1, shuffle=False, collate_fn=collate_fn,
                )
        else:
            # Legacy inline Dirichlet (kept for back-compat smoke tests)
            split = dirichlet_split_patients(all_pids, num_hospitals, alpha, seed=seed)
            split.bind_to_dataset(test_ds)
            print(summarize_split(split))
            rng = random.Random(seed)
            for hid, indices in split.indices_per_hospital.items():
                if not indices:
                    continue
                idxs = list(indices)
                rng.shuffle(idxs)
                n_eval = max(1, int(len(idxs) * 0.2))
                train_idx = idxs[n_eval:]
                hospital_train_loaders[hid] = build_hospital_train_loader(
                    test_ds, train_idx, collate_fn,
                )
                hospital_patient_streams[hid] = build_patient_streams(
                    test_ds, split.pids_per_hospital[hid], collate_fn,
                    max_patients=max_patients_per_hospital,
                )
                print(f"  hosp {hid}: cohort_train={len(train_idx)}, patient_streams="
                      f"{len(hospital_patient_streams[hid])}")
            rng_eval = random.Random(seed)
            for hid, ld in hospital_train_loaders.items():
                eval_idxs = list(ld.dataset.indices)
                rng_eval.shuffle(eval_idxs)
                hospital_eval_loaders[hid] = DataLoader(
                    Subset(test_ds, eval_idxs[:eval_per_hosp_cfg]),
                    batch_size=1, shuffle=False, collate_fn=collate_fn,
                )

        coord_cfg = CoordinatorConfig(
            cohort_cfg=CohortConfig(
                bottleneck=cohort_bottleneck, lr=cohort_lr,
                local_epochs=cohort_local_epochs,
                local_steps_per_epoch=cohort_local_steps,
                objective="downstream",
            ),
            patient_cfg=PatientConfig(bottleneck=patient_bottleneck, lr=patient_lr,
                                       n_iters=n_patient_iters),
            patients_per_round=patients_per_round,
            population_update_freq=population_update_freq,
            eta_cohort=eta_cohort,
            eta_population=eta_population,
        )
        coordinator = HierarchicalCoordinator(
            population_clf, hospital_train_loaders, hospital_patient_streams,
            label_fn, device, cfg=coord_cfg,
        )

        # Baseline AUC: population-only, no adapters
        pop_eval_indices = sorted({
            idx for ld in hospital_train_loaders.values()
            for idx in ld.dataset.indices
        })[:n_pop_eval]
        pop_eval_loader = DataLoader(Subset(test_ds, pop_eval_indices), batch_size=1,
                                      shuffle=False, collate_fn=collate_fn)
        baseline = eval_auc(population_clf, pop_eval_loader, label_fn, device)
        print(f"  baseline (population-only) global AUC = {baseline.auc}")
        all_rows.append({"task": task_name, "round": 0, "phase": "baseline",
                         "global_auc": baseline.auc})
        pd.DataFrame(all_rows).to_csv(output_dir / "hierarchical_history.csv", index=False)

        # Run rounds
        for r in range(1, rounds + 1):
            log = coordinator.run_round()
            # Evaluate cohort models (no patient adapter) on a held-out per-hospital
            # eval pool — patient-late-only sets were too tiny / single-class.
            for hid in coordinator.cohort_states:
                if hid not in hospital_eval_loaders:
                    continue
                p_eval_loader = hospital_eval_loaders[hid]
                res = evaluate_with_states(
                    population_clf, coordinator.cohort_states[hid], None, p_eval_loader,
                    label_fn, device, cohort_bottleneck, patient_bottleneck,
                )
                log[f"hosp_{hid}_cohort_auc"] = res.auc
            log["task"] = task_name; log["phase"] = "hierarchical"
            all_rows.append(log)
            print(f"  round {r}: {log}")
            pd.DataFrame(all_rows).to_csv(output_dir / "hierarchical_history.csv", index=False)

    print(f"\nWrote {output_dir / 'hierarchical_history.csv'}")
    print(pd.DataFrame(all_rows).to_string(index=False))


if __name__ == "__main__":
    main()
