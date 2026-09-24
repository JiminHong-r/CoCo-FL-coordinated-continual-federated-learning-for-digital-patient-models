"""Does adaptation beat simply recalibrating the population model?

Adaptation lowers calibration error, but there is a simple explanation to rule out
first. The population head is fitted class-balanced and evaluated at natural
prevalence, so it is over-confident at every centre, heterogeneous or not, and any
local fine-tuning partly undoes that. So the adapted methods are compared with two
baselines::

    raw           the population model as reported
    global-recal  one intercept for all centres (removes the balanced-fit mismatch)
    centre-recal  one intercept per centre (also removes per-centre label shift)

``centre-recal`` costs one number per centre, while a cohort adapter has 66,112
parameters. If adaptation does not beat it, the adapters add nothing on top of a
per-centre intercept.

The intercepts are fitted out of fold (K folds over windows; fit on the other folds,
apply to the held-out one), so no window helps fit the correction it is scored with.
Recalibration is monotone, so AUROC does not change; it is printed as a check.

::

    python scripts/recalibration_baseline.py <cell_dir> [--folds 5] [--n 300]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import load, macro_metric, reportable  # noqa: E402

METHODS = ["local_finetune", "standard_fedavg", "top_down_only", "hierarchical"]
REF = "population_only"
EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _fit_intercept(logits: np.ndarray, y: np.ndarray) -> float:
    """Intercept ``b`` minimising the NLL of ``sigmoid(logit + b)``, slope fixed at 1.

    Fixing the slope keeps this the smallest correction: a pure shift in baseline
    risk, the right response to a base-rate change and exactly what a frozen head
    cannot do.
    """
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.0
    lo, hi = -12.0, 12.0
    for _ in range(200):                       # bisection on dNLL/db
        mid = 0.5 * (lo + hi)
        grad = float(np.mean(1.0 / (1.0 + np.exp(-(logits + mid))) - y))
        if grad > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def recalibrate(per_hosp: dict, mode: str, folds: int, seed: int = 0) -> dict:
    """Out-of-fold recalibrated predictions. ``mode`` is 'global' or 'centre'."""
    rng = np.random.default_rng(seed)
    hosps = sorted(per_hosp)
    assign = {h: rng.integers(0, folds, len(per_hosp[h][0])) for h in hosps}
    out = {h: (per_hosp[h][0].astype(float).copy(), per_hosp[h][1]) for h in hosps}
    for k in range(folds):
        if mode == "global":
            tr_l = np.concatenate([_logit(per_hosp[h][0])[assign[h] != k] for h in hosps])
            tr_y = np.concatenate([per_hosp[h][1][assign[h] != k] for h in hosps])
            b = _fit_intercept(tr_l, tr_y)
            for h in hosps:
                m = assign[h] == k
                out[h][0][m] = 1.0 / (1.0 + np.exp(-(_logit(per_hosp[h][0])[m] + b)))
        else:
            for h in hosps:
                tr = assign[h] != k
                b = _fit_intercept(_logit(per_hosp[h][0])[tr], per_hosp[h][1][tr])
                m = assign[h] == k
                out[h][0][m] = 1.0 / (1.0 + np.exp(-(_logit(per_hosp[h][0])[m] + b)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell_dir")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n", type=int, default=300, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = load(args.cell_dir)
    if REF not in data:
        sys.exit(f"{REF} not in {sorted(data)}")
    kept, dropped = reportable(data[REF], 10, 10)
    hosps = [h for h, _, _ in kept]
    if not hosps:
        sys.exit("no centre clears the evidence floor")
    data = {m: {h: v for h, v in d.items() if h in hosps} for m, d in data.items()}

    print(f"{os.path.basename(args.cell_dir)}")
    print(f"  centres {hosps}" + (f"   excluded {[h for h, _, _ in dropped]}" if dropped else ""))
    print(f"  {args.folds}-fold out-of-fold intercept fitting\n")

    cand = dict(data)
    cand["POP + global-recal"] = recalibrate(data[REF], "global", args.folds)
    cand["POP + centre-recal"] = recalibrate(data[REF], "centre", args.folds)

    order = [REF, "POP + global-recal", "POP + centre-recal"] + \
            [m for m in METHODS if m in data]
    ece = {k: macro_metric(cand[k], metric="ece") for k in order}
    auroc = {k: macro_metric(cand[k], metric="auroc") for k in order}

    print(f"  {'model':<24}{'macro ECE':>11}{'vs raw':>10}{'macro AUROC':>13}")
    for k in order:
        d = "" if k == REF else f"{ece[k] - ece[REF]:+.4f}"
        print(f"  {k:<24}{ece[k]:>11.4f}{d:>10}{auroc[k]:>13.5f}")

    # The key comparison: adaptation against a per-centre intercept.
    base = "POP + centre-recal"
    n_by = {h: len(cand[REF][h][0]) for h in hosps}
    rng = np.random.default_rng(args.seed)
    idxs = [{h: rng.integers(0, n_by[h], n_by[h]) for h in hosps} for _ in range(args.n)]
    ref_vals = [macro_metric(cand[base], ix, "ece") for ix in idxs]

    print(f"\n  paired ECE difference vs `{base}`  (negative = adaptation better)")
    print(f"  {'method':<24}{'Δ ECE':>10}{'95% CI on Δ':>24}{'P(better)':>11}")
    for m in [x for x in METHODS if x in data]:
        diffs = []
        for ix, r in zip(idxs, ref_vals):
            a = macro_metric(cand[m], ix, "ece")
            if a is not None and r is not None:
                diffs.append(a - r)
        if not diffs:
            continue
        d = np.asarray(diffs)
        lo, hi = np.percentile(d, [2.5, 97.5])
        verdict = "  * adaptation better" if hi < 0 else (
            "  * recal better" if lo > 0 else "")
        print(f"  {m:<24}{ece[m] - ece[base]:>+10.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>24}{(d < 0).mean():>11.3f}{verdict}")

    print("\n  A per-centre intercept is ONE scalar; a cohort adapter is 66,112")
    print("  parameters. Any claim for adaptation has to clear this row.")


if __name__ == "__main__":
    main()
