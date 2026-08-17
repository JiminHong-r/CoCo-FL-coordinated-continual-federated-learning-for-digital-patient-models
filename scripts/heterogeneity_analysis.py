"""Is there enough centre-specific signal to justify partitioning the training data?

WHY THIS EXISTS
---------------
The characterisation so far measures ONE kind of heterogeneity: label shift, i.e.
per-centre differences in P(Y). That is the least demanding kind, and it is not the
kind personalisation needs.

Personalisation pays only if the *decision function* differs by centre. Partitioning
the training signal three ways is a real cost — each cohort adapter is estimated
from a third of the data, so its variance rises — and that cost is only recovered
if per-centre structure exists to be learned. The question is therefore not "are
the centres different?" but "are they different in a way that exceeds the variance
penalty of splitting the data to exploit it?"

This script decomposes the question into the three shifts, and prices the variance.

  P(X)   COVARIATE shift    do centres occupy different regions of score space?
  P(Y|X) CONCEPT shift      does the optimal mapping from score to risk differ?
  P(Y)   LABEL shift        do base rates differ?               (already measured)

CONCEPT SHIFT IS THE DECISIVE ONE. If P(Y|X) is common across centres, then no
per-centre adapter can beat a shared model regardless of how different P(X) or
P(Y) are — the centres are drawing from one conditional law and pooling is
strictly better, because pooling estimates that law from three times the data.

Working in the population model's SCORE space is the right resolution for this
question, not a limitation. The cohort adapter is a residual MLP on the final
hidden state feeding a frozen head, so everything it can express reduces to a
monotone-or-not reshaping of that score. If P(Y | score) is common across centres,
the adapter has nothing to condition on — which is a statement about the
architecture's capacity meeting the data's structure, exactly the framing the
supervisor asked for.

Uses only saved predictions: no GPU, no retraining.

    python scripts/heterogeneity_analysis.py <cell_dir> [<cell_dir> ...]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import load  # noqa: E402

REF = "population_only"
EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _platt(scores: np.ndarray, y: np.ndarray, iters: int = 200) -> tuple[float, float]:
    """Fit y ~ sigmoid(a * logit(score) + b) by Newton steps on (a, b).

    The pair (a, b) IS the centre's decision function in score space: `b` is its
    prior/threshold and `a` how sharply score maps to risk. Comparing (a, b) across
    centres is therefore a direct test for concept shift, and it is exactly the
    correction a per-centre model could apply for free.
    """
    x = _logit(scores)
    a, b = 1.0, 0.0
    for _ in range(iters):
        z = a * x + b
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(p * (1.0 - p), 1e-9, None)
        r = y - p
        # Newton step on the 2-parameter logistic
        h11 = float((w * x * x).sum()); h12 = float((w * x).sum())
        h22 = float(w.sum())
        g1 = float((r * x).sum()); g2 = float(r.sum())
        det = h11 * h22 - h12 * h12
        if abs(det) < 1e-12:
            break
        da = (h22 * g1 - h12 * g2) / det
        db = (h11 * g2 - h12 * g1) / det
        a += da; b += db
        if max(abs(da), abs(db)) < 1e-9:
            break
    return a, b


def _auroc_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil standard error — the price of estimating from fewer samples."""
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2)
           + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS distance between score distributions."""
    grid = np.union1d(a, b)
    ca = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def analyse(cell: str) -> None:
    data = load(cell)
    if REF not in data:
        print(f"{os.path.basename(cell)}: no {REF}")
        return
    per = data[REF]
    hosps = sorted(per)
    print(f"\n{'=' * 78}\n{os.path.basename(cell)}\n{'=' * 78}")

    # --- P(Y): label shift, and the variance price of partitioning -----------
    print("  P(Y) label shift, and the estimation cost of splitting the signal")
    print(f"    {'centre':<10}{'n':>9}{'pos':>8}{'prev':>9}{'AUROC':>9}{'SE':>9}")
    tot_p, tot_n, aucs = 0, 0, []
    from sklearn.metrics import roc_auc_score
    for h in hosps:
        p, y = per[h]
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        se = _auroc_se(auc, n_pos, n_neg) if n_pos and n_neg else float("nan")
        print(f"    h{h:<9}{len(y):>9}{n_pos:>8}{y.mean():>9.4f}{auc:>9.4f}{se:>9.4f}")
        tot_p += n_pos; tot_n += n_neg; aucs.append(auc)
    pooled = np.concatenate([per[h][0] for h in hosps])
    pooled_y = np.concatenate([per[h][1] for h in hosps])
    pooled_auc = roc_auc_score(pooled_y, pooled)
    pooled_se = _auroc_se(pooled_auc, tot_p, tot_n)
    mean_se = float(np.nanmean([
        _auroc_se(roc_auc_score(per[h][1], per[h][0]), int(per[h][1].sum()),
                  int(len(per[h][1]) - per[h][1].sum()))
        for h in hosps if len(np.unique(per[h][1])) > 1]))
    print(f"    {'POOLED':<10}{tot_p + tot_n:>9}{tot_p:>8}"
          f"{tot_p / (tot_p + tot_n):>9.4f}{pooled_auc:>9.4f}{pooled_se:>9.4f}")
    print(f"    variance price of partitioning: mean per-centre SE {mean_se:.4f} "
          f"vs pooled {pooled_se:.4f}  ->  {mean_se / pooled_se:.2f}x")

    # --- P(X): covariate shift in score space -------------------------------
    print("\n  P(X) covariate shift: KS distance between centres' score distributions")
    ks_vals = []
    for i, hi in enumerate(hosps):
        for hj in hosps[i + 1:]:
            d = _ks(per[hi][0], per[hj][0])
            ks_vals.append(d)
            print(f"    h{hi} vs h{hj}: KS = {d:.4f}")

    # --- P(Y|X): concept shift, the decisive quantity ------------------------
    print("\n  P(Y|X) CONCEPT shift: per-centre decision function in score space")
    print("    y ~ sigmoid(a * logit(score) + b);  a=1, b=0 would be 'shared model")
    print("    is already optimal for this centre'")
    print(f"    {'centre':<10}{'slope a':>10}{'intercept b':>13}")
    coefs = {}
    for h in hosps:
        p, y = per[h]
        if len(np.unique(y)) < 2:
            continue
        a, b = _platt(p, y)
        coefs[h] = (a, b)
        print(f"    h{h:<9}{a:>10.4f}{b:>13.4f}")
    a_all, b_all = _platt(pooled, pooled_y)
    print(f"    {'POOLED':<10}{a_all:>10.4f}{b_all:>13.4f}")

    if len(coefs) > 1:
        a_sp = max(a for a, _ in coefs.values()) - min(a for a, _ in coefs.values())
        b_sp = max(b for _, b in coefs.values()) - min(b for _, b in coefs.values())
        print(f"\n    spread across centres: slope {a_sp:.4f}, intercept {b_sp:.4f}")

        # What does a per-centre decision function BUY, out of fold? If concept
        # shift is real, applying each centre's own (a, b) beats the pooled one.
        rng = np.random.default_rng(0)
        gain_own, gain_pooled = [], []
        for h in hosps:
            p, y = per[h]
            if len(np.unique(y)) < 2:
                continue
            fold = rng.integers(0, 5, len(y))
            own = np.empty(len(y)); pool = np.empty(len(y))
            for k in range(5):
                tr, te = fold != k, fold == k
                if len(np.unique(y[tr])) < 2:
                    own[te] = p[te]; pool[te] = p[te]; continue
                ah, bh = _platt(p[tr], y[tr])
                own[te] = 1 / (1 + np.exp(-(ah * _logit(p[te]) + bh)))
                ap, bp = _platt(pooled, pooled_y)
                pool[te] = 1 / (1 + np.exp(-(ap * _logit(p[te]) + bp)))
            gain_own.append(_nll(own, y)); gain_pooled.append(_nll(pool, y))
        if gain_own:
            print(f"    out-of-fold NLL, own decision function {np.mean(gain_own):.5f} "
                  f"vs pooled {np.mean(gain_pooled):.5f}  ->  "
                  f"{np.mean(gain_pooled) - np.mean(gain_own):+.5f}")
            print("    (positive = a per-centre function helps; this is the ENTIRE")
            print("     budget available to any method that reshapes the score)")


def _nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+")
    args = ap.parse_args()
    for c in args.cells:
        if not glob.glob(os.path.join(c, "predictions_*.npz")):
            print(f"{c}: no predictions, skipped")
            continue
        analyse(c)


if __name__ == "__main__":
    main()
