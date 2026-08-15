"""WP2 task 2+3: Dirichlet α sweep characterization.

For each α ∈ {0.1, 0.5, 1.0, 10.0} (configurable):
  1. Partition patients into N hospitals via Dirichlet
  2. Per hospital: patient count, sequence count, per-task positive rate, length stats
  3. Aggregate across α values into a CSV + markdown report

Inputs:
  --tokenized-dir : ETHOS-tokenised data dir (contains static_data.pickle)
  --alphas        : comma-separated α values (default 0.1,0.5,1.0,10.0)
  --num-hospitals : default 3
  --seed          : default 42
  --out-dir       : where to write artefacts

Outputs in ``out-dir``:
  - wp2_alpha_sweep.csv      : one row per (alpha, hospital, task)
  - wp2_alpha_summary.csv    : one row per (alpha, hospital) aggregate stats
  - wp2_alpha_report.md      : human-readable summary
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Import the existing split + dataset helpers.
from cocofl.constants import SpecialToken as ST
from cocofl.constants import horizon_microseconds
from cocofl.data import HospitalMortalityDataset, ICUAdmissionDataset, ReadmissionDataset
from cocofl.downstream.probe import window_labels
from cocofl.evaluation.batching import sequence_lengths
from cocofl.federated.fixed_splits import load_split
from cocofl.federated.hospital_split import (
    HospitalSplit,
    dirichlet_split_patients,
    summarize_split,
)

# Labels are identical for every hospital of every alpha, so compute them once
# per (dataset, task) instead of once per cell.
_LABEL_CACHE: dict[tuple[int, str], "np.ndarray"] = {}

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


def positive_rate(dataset, indices, pos_tokens, task_name=None) -> tuple[int, int]:
    """Return (n_positive, n_total) for the given indices.

    Reads labels straight off the dataset's index tensors (``window_labels``)
    instead of calling ``dataset[idx]``, which builds a full 2048-token window
    per index just to read one field. That cost is why this characterisation
    was capped at 2,000 windows per cell; vectorised, the whole partition is
    affordable and the reported positive rates are exact rather than a
    2,000-sample estimate.

    The label horizon is applied, so these rates match what the probes and the
    WP5 harness actually train against (30 days for readmission).
    """
    key = (id(dataset), task_name or "")
    if key not in _LABEL_CACHE:
        _LABEL_CACHE[key] = window_labels(
            dataset, pos_tokens, horizon_microseconds(task_name) if task_name else None,
        )
    y = _LABEL_CACHE[key]
    idx = np.asarray(list(indices), dtype=np.int64)
    if idx.size == 0:
        return 0, 0
    return int(y[idx].sum()), int(idx.size)


def sequence_length_stats(dataset, indices) -> dict:
    """Return min/max/mean/median sequence length over indices.

    Uses ``evaluation.batching.sequence_lengths``, which derives the exact
    length ``__getitem__`` would produce from the index tensors. The previous
    version materialised each window and was hardcoded to the first 2,000
    indices, so these stats described a prefix rather than the partition.
    """
    idx = list(indices)
    if not idx:
        return {"n": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    lengths = sequence_lengths(dataset, idx)
    if lengths is None:  # dataset cannot expose its layout — fall back
        lengths = []
        for i in idx[:2000]:
            try:
                x, _ = dataset[i]
                tokens = x[0] if isinstance(x, tuple) else x
                lengths.append(int(tokens.size(-1)))
            except Exception:
                continue
    else:
        lengths = lengths.tolist()
    if len(lengths) == 0:
        return {"n": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    arr = np.asarray(lengths)
    return {"n": len(arr), "min": int(arr.min()), "max": int(arr.max()),
            "mean": float(arr.mean()), "median": float(np.median(arr))}


def temporal_density_stats(dataset, indices) -> dict:
    """Distribution of inter-event time deltas across patient timelines.

    Spec WP2 task 3: 'temporal density' characterises how dense events are
    along the timeline. Uses the dataset's underlying ``times`` tensor
    (microseconds since epoch) — computes deltas (hours) per index and
    aggregates over up to 2000 indices for speed.

    Returns p25/p50/p75 of inter-event hours plus a mean events-per-day rate.
    """
    base = dataset if hasattr(dataset, "times") else getattr(dataset, "dataset", dataset)
    starts = getattr(base, "start_indices", None)
    if starts is None:
        return {"n_deltas": 0, "delta_p25_h": 0.0, "delta_p50_h": 0.0,
                "delta_p75_h": 0.0, "events_per_day": 0.0}
    deltas_hr: list[float] = []
    # A 32-token lookback is degenerate here: the tokens immediately preceding a
    # prediction point all belong to the same admission event and share one
    # timestamp, so every delta is 0. 512 spans a real slice of the patient
    # history the model conditions on (context is 2048).
    lookback = 512
    # ``indices`` are SAMPLE indices; ``times`` is indexed by TOKEN. Reading
    # times[sample_idx] measured an unrelated stretch of the corpus — the same
    # sample-vs-token confusion that broke the hospital partition. Map through
    # start_indices, and look BACK from the prediction point, which is the
    # history the model actually conditions on.
    for i in indices[:20000]:
        try:
            tok = int(starts[int(i)])
            lo = max(0, tok - lookback + 1)
            t = base.times[lo : tok + 1]
            if t.numel() < 2:
                continue
            d = (t[1:].float() - t[:-1].float()) / 3_600_000_000.0
            d = d[d >= 0]
            deltas_hr.extend(d.cpu().tolist())
        except Exception:
            continue
    if not deltas_hr:
        return {"n_deltas": 0, "delta_p25_h": 0.0, "delta_p50_h": 0.0,
                "delta_p75_h": 0.0, "events_per_day": 0.0}
    arr = np.asarray(deltas_hr)
    # Many adjacent tokens share the same timestamp (events recorded at the
    # same admission moment). Use the mean over POSITIVE deltas so the rate
    # reflects real inter-event spacing, not within-event batching.
    arr_pos = arr[arr > 0]
    if arr_pos.size == 0:
        mean_h = float(arr.mean())
    else:
        mean_h = float(arr_pos.mean())
    rate_per_day = (24.0 / mean_h) if mean_h > 0 else 0.0
    return {
        "n_deltas": int(arr.size),
        "delta_p25_h": float(np.percentile(arr, 25)),
        "delta_p50_h": float(np.median(arr)),
        "delta_p75_h": float(np.percentile(arr, 75)),
        "events_per_day": rate_per_day,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenized-dir", required=True, type=Path)
    p.add_argument("--splits-dir", type=Path, default=None,
                   help="dir with fixed federated splits (alpha_*.json from "
                        "build_split_v2.py). When set, characterise the ACTUAL "
                        "centre assignments (50%% federated pool, global-test and "
                        "population reserve excluded) instead of re-drawing "
                        "Dirichlet over the whole tokenized dir.")
    p.add_argument("--alphas", default="0.1,0.5,1.0,10.0",
                   help="comma-separated Dirichlet concentrations")
    p.add_argument("--num-hospitals", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--task-eval-cap", type=int, default=10**9,
                   help="max samples per (hospital, task) for positive-rate stats")
    args = p.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load static_data once for the patient ID list
    with open(args.tokenized_dir / "static_data.pickle", "rb") as f:
        static_data = pickle.load(f)
    all_pids = list(static_data.keys())
    print(f"Total patients in {args.tokenized_dir}: {len(all_pids):,}")

    # Build each task's dataset ONCE so we can re-use across α (datasets are large)
    print("Loading task datasets ...")
    task_ds = {name: cls(args.tokenized_dir) for name, cls in TASKS.items()}
    for name, ds in task_ds.items():
        print(f"  {name}: {len(ds):,} samples")

    full_rows: list[dict] = []
    summary_rows: list[dict] = []

    for alpha in alphas:
        print(f"\n=== α = {alpha} ===")
        if args.splits_dir is not None:
            # Characterise the ACTUAL centre assignments from the fixed split,
            # i.e. only the 50% federated pool (global-test + population reserve
            # excluded). Each centre's patient set is train ∪ val ∪ test.
            split_json = load_split(args.splits_dir, alpha)
            pids_per_hospital = {
                int(hid): list(parts["train"]) + list(parts["val"]) + list(parts["test"])
                for hid, parts in split_json["hospitals"].items()
            }
            split = HospitalSplit(
                num_hospitals=split_json["num_hospitals"],
                alpha=alpha,
                pids_per_hospital=pids_per_hospital,
            )
        else:
            split = dirichlet_split_patients(all_pids, args.num_hospitals, alpha, seed=args.seed)
        # Fraction is relative to the partitioned pool (the 50% federated pool
        # when fixed splits are used, else the whole tokenized dir).
        total_for_frac = sum(len(p) for p in split.pids_per_hospital.values())
        # Bind to one of the datasets first to fill indices_per_hospital
        # (We use the first task's dataset; the patient set is the same across tasks,
        # but per-task index lists differ.)
        for hid, pids in split.pids_per_hospital.items():
            n_pids = len(pids)
            row_base = {"alpha": alpha, "hospital": hid, "n_patients": n_pids}
            summary = {**row_base, "frac_patients": n_pids / max(total_for_frac, 1)}

            for task_name, ds in task_ds.items():
                split.bind_to_dataset(ds)
                idxs = split.indices_per_hospital.get(hid, [])[:args.task_eval_cap]
                n_pos, n_tot = positive_rate(ds, idxs, POSITIVE_TOKENS[task_name], task_name)
                pos_rate = (n_pos / n_tot) if n_tot else None
                full_rows.append({**row_base, "task": task_name,
                                  "n_samples_eval": n_tot, "n_positive": n_pos,
                                  "positive_rate": pos_rate})
                # Keep per-task pos_rate in summary too
                summary[f"{task_name}_pos_rate"] = pos_rate
                summary[f"{task_name}_n_samples"] = n_tot

            # Sequence-length stats (use the first task's dataset for tokens shape)
            split.bind_to_dataset(task_ds["hospital_mortality"])
            idxs = split.indices_per_hospital.get(hid, [])
            seq_stats = sequence_length_stats(task_ds["hospital_mortality"], idxs)
            summary["seq_n"] = seq_stats["n"]
            summary["seq_min"] = seq_stats["min"]
            summary["seq_max"] = seq_stats["max"]
            summary["seq_mean"] = round(seq_stats["mean"], 1)
            summary["seq_median"] = seq_stats["median"]
            # Temporal density (Spec WP2 task 3 — inter-event time per centre)
            td = temporal_density_stats(task_ds["hospital_mortality"], idxs)
            summary["delta_p25_h"] = round(td["delta_p25_h"], 3)
            summary["delta_p50_h"] = round(td["delta_p50_h"], 3)
            summary["delta_p75_h"] = round(td["delta_p75_h"], 3)
            summary["events_per_day"] = round(td["events_per_day"], 2) if td["events_per_day"] != float("inf") else "inf"
            summary_rows.append(summary)
            print(f"  hosp {hid}: {n_pids:,} patients, seq_mean={seq_stats['mean']:.0f}, "
                  f"events/day={summary['events_per_day']}, "
                  f"task_pos_rates="
                  f"{ {t: round(summary.get(f'{t}_pos_rate') or 0, 4) for t in TASKS} }")

    full_df = pd.DataFrame(full_rows)
    summary_df = pd.DataFrame(summary_rows)
    full_df.to_csv(args.out_dir / "wp2_alpha_sweep.csv", index=False)
    summary_df.to_csv(args.out_dir / "wp2_alpha_summary.csv", index=False)

    # Markdown report
    lines = [f"# WP2 Dirichlet α-sweep characterization\n",
             f"- Tokenized dir: `{args.tokenized_dir}`",
             f"- Total patients: {len(all_pids):,}",
             f"- Hospitals: {args.num_hospitals}",
             f"- α values: {alphas}",
             f"- Seed: {args.seed}\n",
             "## Summary (per hospital, per α)\n",
             summary_df.to_markdown(index=False),
             "\n\n## Per-task positive rates (per hospital, per α)\n",
             full_df.to_markdown(index=False),
             ""]
    (args.out_dir / "wp2_alpha_report.md").write_text("\n".join(lines))
    print(f"\nWrote artefacts to {args.out_dir}")


if __name__ == "__main__":
    main()
