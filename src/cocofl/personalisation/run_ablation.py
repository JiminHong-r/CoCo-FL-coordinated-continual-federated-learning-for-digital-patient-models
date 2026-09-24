"""Personalisation ablation runner for TFN + FFN-adapter pipeline (WP3).

Per-patient mode (default): for each held-out patient, chronologically split
the patient's samples early/late, copy the population classifier, personalise
on the early half, predict on the late half. Predictions pooled across patients
for personal AUC. Population retention measured on patients NOT in the cohort.

Strategies covered:
    A0_full_ft        — full fine-tune
    A1_head_only      — head only
    A2_ffn            — FFN adapter (Houlsby) + head
    A3_ffn_ewc        — FFN + EWC anchor
    A4_ffn_replay     — FFN + population replay
    A5_ffn_distill    — FFN + population distillation

Usage::
    cocofl_ablation_tfn \\
        model_path=/path/to/best_model.pt \\
        train_dir=/path/to/mimic-train \\
        input_dir=/path/to/mimic-test \\
        output_dir=wp3_artifacts \\
        +n_patients=20 +n_iters=200 +n_pop_eval=2000 +pop_probe_patients=3
"""
from __future__ import annotations

import copy
import gc
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from ..constants import SpecialToken as ST
from ..constants import STATIC_DATA_FN, horizon_microseconds
from ..data import HospitalMortalityDataset, ICUAdmissionDataset, ReadmissionDataset
from ..data.static_features import build_static_arrays, NUMERICAL_DIM
from ..models.classifier import TFNDownstreamClassifier
from ..models.backbone import TFNFullModel
from ..vocabulary import Vocabulary
from ..adapters import attach_ffn_adapter, count_trainable_params
from .continual import (
    DistillationLossTFN, EWCRegularizerTFN, ReplayBufferTFN,
    clone_frozen_classifier, replay_loss_tfn,
)
from .evaluate import auc_from_preds, eval_auc, predict
from .loop import (
    PersonalizeConfig, per_patient_subsets, personalize_full,
    personalize_head_only, personalize_ffn, split_indices_by_time,
)


POSITIVE_TOKENS = {
    "hospital_mortality": {ST.DEATH},
    "readmission": {ST.ADMISSION},
    "icu_admission": {ST.ICU_ADMISSION, ST.DEATH},
}
TASKS = {
    "hospital_mortality": HospitalMortalityDataset,
    "readmission": ReadmissionDataset,
    "icu_admission": ICUAdmissionDataset,
}

STRATEGIES = (
    "A0_full_ft", "A1_head_only", "A2_ffn",
    "A3_ffn_ewc", "A4_ffn_replay", "A5_ffn_distill",
)


# ---- collate + label fns (TFN-format batches) -------------------------------

def make_collate(cat_by_pid, num_by_pid, dataset=None):
    """Collate for the WP4/WP5 harness.

    ``dataset`` must be supplied to get REAL inter-event times to the T-LSTM. It
    used to be absent, and elapsed was synthesised as all-ones, which collapsed
    the time-decay to a constant and ran the backbone outside its training
    distribution — see cocofl.data.elapsed for the measurements. Falling back is
    still possible (some ablation entry points build a collate before the dataset
    exists) but it WARNS, because a silent fallback here is the original defect.
    """
    from ..data.elapsed import dataset_supports_elapsed, elapsed_for_window

    use_real = dataset is not None and dataset_supports_elapsed(dataset)
    if dataset is not None and not use_real:
        import warnings
        warnings.warn(
            "make_collate: dataset cannot supply inter-event times; falling back "
            "to elapsed=1.0, which disables the T-LSTM time decay (WP3 task 1).",
            stacklevel=2,
        )

    def collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        xs = torch.stack([b[0] for b in batch])
        pids = [b[1].get("patient_id") for b in batch]
        cats = torch.stack([cat_by_pid.get(pid, torch.zeros(3, dtype=torch.long)) for pid in pids])
        nums = torch.stack([num_by_pid.get(pid, torch.zeros(NUMERICAL_DIM, dtype=torch.float32)) for pid in pids])
        if use_real:
            # Reconstruct each window's real Δt (hours) from the token index the
            # dataset already records, so the T-LSTM sees what it was trained on.
            elapsed = torch.stack([
                elapsed_for_window(dataset, b[1]["data_idx"], xs.size(1))
                for b in batch
            ])
        else:
            elapsed = torch.ones_like(xs, dtype=torch.float32)
        labels = {
            "expected": [b[1]["expected"] for b in batch],
            "patient_id": [b[1].get("patient_id") for b in batch],
            "prediction_time": [b[1].get("prediction_time") for b in batch],
            # Needed by time-bounded tasks (see constants.TASK_HORIZONS): elapsed
            # microseconds from the prediction point to the outcome token.
            "true_token_time": [b[1].get("true_token_time") for b in batch],
        }
        return (xs, elapsed, cats, nums), labels
    return collate


def make_label_fn(task_name: str):
    """Binary label fn for ``task_name``.

    For tasks with a horizon in ``constants.TASK_HORIZONS`` (e.g. readmission =
    30 days) the outcome token must ALSO fall within that horizon; a qualifying
    event later than the horizon counts as a negative.
    """
    pos = POSITIVE_TOKENS[task_name]
    horizon_us = horizon_microseconds(task_name)
    if horizon_us is None:
        return lambda y: [1 if e in pos else 0 for e in y["expected"]]

    def label_fn(y):
        times = y.get("true_token_time")
        if times is None:
            raise KeyError(
                f"task {task_name!r} has a {horizon_us / 8.64e10:.0f}-day label horizon "
                "but the batch carries no 'true_token_time'; the collate must "
                "forward it (see make_collate)."
            )
        return [
            1 if (e in pos and t is not None and 0 <= t <= horizon_us) else 0
            for e, t in zip(y["expected"], times)
        ]

    return label_fn


def make_loader(dataset, indices, collate_fn, batch_size: int = 1, shuffle: bool = True):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
    )


# ---- model loading ----------------------------------------------------------

def build_population_classifier(model_path: str | Path, device: torch.device,
                                 num_classes: int = 1) -> TFNDownstreamClassifier:
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
    return TFNDownstreamClassifier(backbone, num_classes=num_classes).to(device)


def offline_head_finetune(classifier, train_dataset, train_idx, label_fn, n_iters, lr, device,
                          collate_fn) -> None:
    loader = make_loader(train_dataset, train_idx, collate_fn)
    personalize_head_only(
        classifier, loader, label_fn,
        PersonalizeConfig(n_iters=n_iters, lr=lr, log_every=max(1, n_iters // 4)),
        device,
    )


# ---- strategy dispatch ------------------------------------------------------

def run_strategy(name: str, base_classifier: TFNDownstreamClassifier,
                  train_loader: DataLoader, label_fn, cfg: PersonalizeConfig,
                  device: torch.device, patient_cat: torch.Tensor, patient_num: torch.Tensor,
                  ) -> TFNDownstreamClassifier:
    clf = copy.deepcopy(base_classifier)

    if name == "A0_full_ft":
        personalize_full(clf, train_loader, label_fn, cfg, device)
    elif name == "A1_head_only":
        personalize_head_only(clf, train_loader, label_fn, cfg, device)
    elif name == "A2_ffn":
        personalize_ffn(clf, train_loader, label_fn, cfg, device)
    elif name == "A3_ffn_ewc":
        attach_ffn_adapter(clf.backbone, attr="patient_adapter")
        # Consolidate Fisher on the per-patient train set (small but local context).
        ewc = EWCRegularizerTFN.from_classifier(
            clf, train_loader, label_fn, device, lambda_ewc=1000.0, max_batches=8,
        )
        personalize_ffn(clf, train_loader, label_fn, cfg, device,
                        extra_loss_fn=ewc.as_extra_loss_fn())
    elif name == "A4_ffn_replay":
        attach_ffn_adapter(clf.backbone, attr="patient_adapter")
        buf = ReplayBufferTFN(capacity=256)
        buf.fill_from(train_loader, label_fn)
        personalize_ffn(clf, train_loader, label_fn, cfg, device,
                        extra_loss_fn=replay_loss_tfn(buf, clf, batch_size=1, device=device, alpha=0.5))
    elif name == "A5_ffn_distill":
        attach_ffn_adapter(clf.backbone, attr="patient_adapter")
        teacher = clone_frozen_classifier(base_classifier)
        distill = DistillationLossTFN(
            teacher=teacher, cat=patient_cat.to(device), num=patient_num.to(device),
            alpha=0.3, temperature=2.0,
        )
        personalize_ffn(clf, train_loader, label_fn, cfg, device,
                        extra_loss_fn=distill.as_extra_loss_fn())
    else:
        raise ValueError(f"unknown strategy: {name!r}")
    return clf


# ---- per-patient split ------------------------------------------------------

@dataclass
class PerPatientSplit:
    cohort_pids: list[int]
    per_patient: dict[int, tuple[list[int], list[int]]]
    population_idx: list[int]


def build_per_patient_split(test_dataset, n_patients: int, train_frac: float,
                             min_samples: int = 4, seed: int = 42) -> PerPatientSplit:
    buckets = per_patient_subsets(test_dataset, min_samples=min_samples)
    rng = random.Random(seed)
    cand = list(buckets.keys())
    rng.shuffle(cand)

    per_patient: dict[int, tuple[list[int], list[int]]] = {}
    for pid in cand:
        if len(per_patient) >= n_patients:
            break
        early, late = split_indices_by_time(test_dataset, buckets[pid], train_frac=train_frac)
        if early and late:
            per_patient[pid] = (early, late)
    cohort_pids = list(per_patient.keys())
    cohort_set = set(cohort_pids)
    pop_idx = [idx for pid, idxs in buckets.items() if pid not in cohort_set for idx in idxs]
    return PerPatientSplit(cohort_pids=cohort_pids, per_patient=per_patient,
                           population_idx=pop_idx)


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_per_patient(pop_clf, test_dataset, splits, pop_eval_idx, label_fn, cfg,
                    pop_probe_patients, device, collate_fn, cat_by_pid, num_by_pid,
                    task_name=None, output_dir=None,
                    accumulated_rows=None, accumulated_pp_rows=None,
                    strategies=STRATEGIES):
    pop_eval_loader = make_loader(test_dataset, pop_eval_idx, collate_fn, shuffle=False)
    baseline_pop = eval_auc(pop_clf, pop_eval_loader, label_fn, device).auc
    print(f"  baseline population AUC = {baseline_pop}")

    rows = [{
        "strategy": "BASELINE", "personal_auc": None, "population_auc": baseline_pop,
        "retention": 1.0, "trainable": 0,
        "total": sum(p.numel() for p in pop_clf.parameters()),
        "n_patients": len(splits.cohort_pids),
    }]
    per_patient_rows: list[dict] = []

    def _save():
        if output_dir is None or accumulated_rows is None:
            return
        all_rows = accumulated_rows + [{**r, "task": task_name, "mode": "per_patient"} for r in rows]
        pd.DataFrame(all_rows).to_csv(output_dir / "ablation_tfn.csv", index=False)
        if accumulated_pp_rows is not None:
            all_pp = accumulated_pp_rows + [{**r, "task": task_name} for r in per_patient_rows]
            pd.DataFrame(all_pp).to_csv(output_dir / "ablation_tfn_per_patient.csv", index=False)

    _save()
    for strat in strategies:
        print(f"\n  --- {strat} ---")
        all_preds: list[float] = []
        all_labels: list[int] = []
        pop_aucs: list[float] = []
        trainable, total = 0, 0
        try:
            for k, pid in enumerate(splits.cohort_pids):
                early, late = splits.per_patient[pid]
                train_loader = make_loader(test_dataset, early, collate_fn, shuffle=True)
                late_loader = make_loader(test_dataset, late, collate_fn, shuffle=False)

                patient_cat = cat_by_pid.get(pid, torch.zeros(3, dtype=torch.long))
                patient_num = num_by_pid.get(pid, torch.zeros(NUMERICAL_DIM, dtype=torch.float32))

                personalised = run_strategy(
                    strat, pop_clf, train_loader, label_fn, cfg, device,
                    patient_cat=patient_cat, patient_num=patient_num,
                )
                preds, labels = predict(personalised, late_loader, label_fn, device)
                all_preds.extend(preds)
                all_labels.extend(labels)

                if k < pop_probe_patients:
                    pa = eval_auc(personalised, pop_eval_loader, label_fn, device).auc
                    if pa is not None:
                        pop_aucs.append(pa)

                trainable, total = count_trainable_params(personalised)
                per_patient_rows.append({
                    "strategy": strat, "patient_id": pid,
                    "n_early": len(early), "n_late": len(late),
                })
                del personalised
                _free_gpu()
        except Exception as e:
            print(f"  !! {strat} FAILED: {type(e).__name__}: {e}")
            rows.append({
                "strategy": strat, "personal_auc": None, "population_auc": None,
                "retention": None, "trainable": 0, "total": 0,
                "n_patients": len(splits.cohort_pids),
                "error": f"{type(e).__name__}: {e}",
            })
            _save()
            continue

        personal = auc_from_preds(all_preds, all_labels)
        pop_auc = float(sum(pop_aucs) / len(pop_aucs)) if pop_aucs else None
        retention = (pop_auc / baseline_pop) if (pop_auc and baseline_pop) else None
        print(f"  pooled personal_auc={personal.auc}  pop_auc={pop_auc}  "
              f"retention={retention}  (n_pos={personal.n_pos}/{personal.n})")
        rows.append({
            "strategy": strat, "personal_auc": personal.auc,
            "population_auc": pop_auc, "retention": retention,
            "trainable": trainable, "total": total,
            "n_patients": len(splits.cohort_pids),
        })
        _save()
    return rows, per_patient_rows


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = Path(cfg.train_dir)
    test_dir = Path(cfg.input_dir)
    model_path = Path(cfg.model_path)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_head_train = int(getattr(cfg, "n_head_train", 1500))
    n_iters = int(getattr(cfg, "n_iters", 200))
    lr = float(getattr(cfg, "lr", 1e-3))
    seed = int(getattr(cfg, "seed", 42))
    n_patients = int(getattr(cfg, "n_patients", 20))
    train_frac = float(getattr(cfg, "train_frac", 0.5))
    min_samples = int(getattr(cfg, "min_samples_per_patient", 4))
    pop_probe_patients = int(getattr(cfg, "pop_probe_patients", 3))
    n_pop_eval = int(getattr(cfg, "n_pop_eval", 2000))

    strat_filter = getattr(cfg, "strategies", None)
    if strat_filter:
        if isinstance(strat_filter, str):
            selected = tuple(s.strip() for s in strat_filter.split(","))
        else:
            selected = tuple(str(s).strip() for s in strat_filter)
        invalid = [s for s in selected if s not in STRATEGIES]
        if invalid:
            raise ValueError(f"unknown strategy in filter: {invalid}")
        strategies = selected
    else:
        strategies = STRATEGIES
    print(f"strategies to run: {strategies}")

    Vocabulary.from_path(test_dir)

    # Build static-feature lookups from train + test directories
    with open(test_dir / STATIC_DATA_FN, "rb") as f:
        test_static = pickle.load(f)
    test_cat, test_num = build_static_arrays(test_static)
    with open(train_dir / STATIC_DATA_FN, "rb") as f:
        train_static = pickle.load(f)
    train_cat, train_num = build_static_arrays(train_static)

    rows: list[dict] = []
    per_patient_rows: list[dict] = []

    for task_name, dataset_cls in TASKS.items():
        print(f"\n=== task: {task_name} (mode=per_patient) ===")
        train_ds = dataset_cls(train_dir)
        test_ds = dataset_cls(test_dir)
        label_fn = make_label_fn(task_name)

        train_collate = make_collate(train_cat, train_num)
        test_collate = make_collate(test_cat, test_num)

        rng = random.Random(seed)
        head_train_idx = rng.sample(range(len(train_ds)), min(n_head_train, len(train_ds)))

        pop_clf = build_population_classifier(model_path, device)
        offline_head_finetune(pop_clf, train_ds, head_train_idx, label_fn, n_iters, lr, device,
                               train_collate)

        ablation_cfg = PersonalizeConfig(
            n_iters=n_iters, lr=lr, log_every=max(1, n_iters // 4),
        )

        splits = build_per_patient_split(
            test_ds, n_patients=n_patients, train_frac=train_frac,
            min_samples=min_samples, seed=seed,
        )
        print(f"  cohort: {len(splits.cohort_pids)} patients, "
              f"{len(splits.population_idx)} population samples available")
        pop_eval_idx = splits.population_idx[:n_pop_eval]

        task_rows, task_pp = run_per_patient(
            pop_clf, test_ds, splits, pop_eval_idx, label_fn, ablation_cfg,
            pop_probe_patients, device, test_collate, test_cat, test_num,
            task_name=task_name, output_dir=output_dir,
            accumulated_rows=rows, accumulated_pp_rows=per_patient_rows,
            strategies=strategies,
        )
        for r in task_rows:
            r["task"] = task_name; r["mode"] = "per_patient"; rows.append(r)
        for r in task_pp:
            r["task"] = task_name; per_patient_rows.append(r)

        pd.DataFrame(rows).to_csv(output_dir / "ablation_tfn.csv", index=False)
        if per_patient_rows:
            pd.DataFrame(per_patient_rows).to_csv(output_dir / "ablation_tfn_per_patient.csv", index=False)

    print("\n=== Final ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
