"""Paired bootstrap on method differences — the correct instrument for "is this a tie?".

WHY THIS EXISTS
---------------
Every WP5 method is scored on IDENTICAL evaluation windows. Their differences are
therefore PAIRED, and a paired difference is estimated far more precisely than the
independent per-method confidence intervals suggest.

The numbers make the point. On class-aware readmission α=0.5 the independent
Hanley-McNeil half-width is about ±0.016, while the observed method differences
are ~0.0006 — a 25× mismatch. Reported that way, every comparison is "inside the
CI" and nothing can ever be resolved, which is not a finding about the methods but
about using the wrong instrument. Resampling the SAME windows for both methods
cancels the shared window-difficulty variance and leaves the interval on the
difference itself.

This is the difference between reporting a tie and demonstrating one.

  python scripts/paired_bootstrap.py results_v9_classaware_readmission_a0.5
  python scripts/paired_bootstrap.py <cell_dir> --ref population_only --n 10000
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

METHODS = ["population_only", "local_finetune", "standard_fedavg",
           "top_down_only", "hierarchical"]


def load(cell_dir: str):
    """{method: {hospital: (pred, label)}} from a cell's predictions npz."""
    files = sorted(glob.glob(os.path.join(cell_dir, "predictions_*.npz")))
    if not files:
        sys.exit(f"no predictions_*.npz in {cell_dir} — the cell predates "
                 f"prediction saving, so a paired test is not possible for it")
    out: dict[str, dict[str, tuple]] = {}
    for f in files:
        z = np.load(f)
        for key in z.files:
            method, hosp, kind = key.split("|")
            if kind != "pred":
                continue
            lab = f"{method}|{hosp}|label"
            if lab not in z.files:
                continue
            out.setdefault(method, {})[hosp] = (z[key], z[lab])
    return out


def reportable(per_hosp: dict, min_pos: int, min_neg: int) -> tuple[list, list]:
    """Split hospitals into (kept, excluded) by the harness's evidence floor.

    WHY THIS MUST MATCH run_evaluation.py
    -------------------------------------
    The CSV's MACRO columns are computed over ``_reportable_hospitals`` — centres
    with at least ``min_pos`` positives and ``min_neg`` negatives. This script
    originally averaged over ALL centres, so the two headline numbers for the same
    cell disagreed, and the disagreement DILUTED a real effect rather than
    inventing one.

    On class-aware mortality α=0.1, h2 has 527 windows and 8 positives, and the
    harness excludes it. Over the floored set ``top_down_only`` is +0.00317 with a
    95 % CI of [+0.00061, +0.00634] — it excludes zero. Averaging h2 back in drags
    the same comparison to +0.00043 with a CI spanning zero, because an AUROC
    estimated from eight positives is nearly pure noise and it carries a third of
    the macro weight. The unfloored version turns a demonstrated win into an
    apparent tie.

    Two aggregates over the same cell must be over the same centres, or one of
    them is wrong.
    """
    kept, dropped = [], []
    for h, (_p, y) in per_hosp.items():
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        (kept if (n_pos >= min_pos and n_neg >= min_neg) else dropped).append(
            (h, n_pos, n_neg))
    return sorted(kept), sorted(dropped)


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error over equal-width bins.

    Mirrors cocofl.evaluation.metrics.expected_calibration_error so the paired
    intervals describe the same quantity the CSV reports. LOWER is better, so a
    NEGATIVE delta in the table below is an improvement.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n, ece = len(p), 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def macro_metric(per_hosp: dict, idx: dict | None = None,
                 metric: str = "auroc") -> float | None:
    """Macro AUROC or AUPRC, optionally on a resample given per hospital.

    AUPRC matters here because it and AUROC have disagreed on every apparent win
    so far: at ~2 % prevalence a method can improve global ranking (AUROC) while
    ranking worse among the high-scoring cases that AUPRC weights. Testing only
    AUROC would report the half of the picture that happens to look better.
    """
    vals = []
    for h, (p, y) in per_hosp.items():
        if idx is not None:
            p, y = p[idx[h]], y[idx[h]]
        if metric == "ece":
            # ECE needs no second class to be defined, but a single-class slice
            # makes it trivially the mean confidence gap; keep the same guard as
            # the rank metrics so all three describe identical centres.
            if len(np.unique(y)) < 2:
                continue
            vals.append(_ece(p, y))
            continue
        if len(np.unique(y)) < 2:
            continue
        score = roc_auc_score if metric == "auroc" else average_precision_score
        vals.append(score(y, p))
    return float(np.mean(vals)) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell_dir")
    ap.add_argument("--ref", default="population_only")
    ap.add_argument("--n", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=42)
    # Defaults mirror the harness's min_eval_pos / min_eval_neg so this script and
    # the CSV MACRO describe the same set of centres.
    ap.add_argument("--min-pos", type=int, default=10)
    ap.add_argument("--min-neg", type=int, default=10)
    ap.add_argument("--no-floor", action="store_true",
                    help="average over ALL centres (will disagree with the CSV MACRO)")
    ap.add_argument("--metric",
                    choices=["auroc", "auprc", "ece", "all"], default="all")
    args = ap.parse_args()

    data = load(args.cell_dir)
    if args.ref not in data:
        sys.exit(f"reference {args.ref} not in {sorted(data)}")

    print(f"{os.path.basename(args.cell_dir)}")
    kept, dropped = reportable(data[args.ref], args.min_pos, args.min_neg)
    if args.no_floor:
        hosps = sorted(data[args.ref])
        print("  evidence floor: DISABLED — this will not match the CSV MACRO")
    else:
        hosps = [h for h, _, _ in kept]
        print(f"  evidence floor (>={args.min_pos} pos, >={args.min_neg} neg): "
              f"MACRO over {hosps}")
        for h, n_pos, n_neg in dropped:
            print(f"    excluded h{h}: pos={n_pos} neg={n_neg}")
        if not hosps:
            sys.exit("  no centre clears the evidence floor — nothing to compare")
    data = {m: {h: v for h, v in per_h.items() if h in hosps}
            for m, per_h in data.items()}
    n_by_h = {h: len(data[args.ref][h][0]) for h in hosps}
    print(f"  windows: {n_by_h}   resamples: {args.n}")

    # Sanity: a paired test is only valid if every method saw the same windows in
    # the same order. Identical lengths AND identical labels is the check.
    for m in data:
        for h in hosps:
            if h not in data[m]:
                continue
            if len(data[m][h][0]) != n_by_h[h] or not np.array_equal(
                    data[m][h][1], data[args.ref][h][1]):
                sys.exit(f"  ABORT: {m} h{h} is not aligned with {args.ref} — "
                         f"the pairing assumption does not hold")
    print("  alignment verified: all methods scored identical windows\n")

    metrics = (["auroc", "auprc", "ece"] if args.metric == "all"
               else [args.metric])
    for metric in metrics:
        # Same seed per metric so both are read off the SAME resamples — the
        # AUROC/AUPRC comparison is then itself paired, not two separate draws.
        rng = np.random.default_rng(args.seed)
        base = {m: macro_metric(data[m], metric=metric) for m in data}
        print(f"\n  {'method':<18}{'macro ' + metric.upper():>12}{'Δ vs ref':>11}"
              f"{'95% CI on Δ':>22}{'P(Δ>0)':>9}")
        for m in METHODS:
            if m not in data:
                continue
            if m == args.ref:
                print(f"  {m:<18}{base[m]:>12.5f}{'—':>11}{'(reference)':>22}{'':>9}")
                continue
            diffs = np.empty(args.n)
            for b in range(args.n):
                # ONE resample of window indices, applied to BOTH methods — this
                # is what cancels the shared per-window difficulty.
                idx = {h: rng.integers(0, n_by_h[h], n_by_h[h]) for h in hosps}
                a = macro_metric(data[m], idx, metric)
                r = macro_metric(data[args.ref], idx, metric)
                diffs[b] = np.nan if (a is None or r is None) else a - r
            d = diffs[~np.isnan(diffs)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            star = "  *" if (lo > 0 or hi < 0) else ""
            if metric == "ece" and star:
                star = "  * better" if hi < 0 else "  * worse"
            print(f"  {m:<18}{base[m]:>12.5f}{base[m]-base[args.ref]:>+11.5f}"
                  f"{f'[{lo:+.5f}, {hi:+.5f}]':>22}{(d > 0).mean():>9.3f}{star}")

    print("\n  * = 95% CI on the paired difference excludes zero.")
    print("  A CI containing zero at this width is a DEMONSTRATED tie, not an")
    print("  unresolved comparison — which is what the independent per-method")
    print("  intervals could never establish.")


if __name__ == "__main__":
    main()
