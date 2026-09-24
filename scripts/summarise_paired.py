"""Paired-bootstrap summary across every testable cell.

One table per metric: the paired difference against ``population_only``, marked when
the 95 % CI excludes zero. One cell cannot show that an effect is consistent; the
table across cells can.

The reference metric depends only on the resample, not on the method, so it is
computed once per resample and reused for every method.

::

    python scripts/summarise_paired.py [--n 300] [--glob "results_v11full*"]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import load, macro_metric, reportable  # noqa: E402

METHODS = ["local_finetune", "standard_fedavg", "top_down_only", "hierarchical"]
REF = "population_only"


def verdict(cell: str, metric: str, n: int) -> dict | None:
    data = load(cell)
    if REF not in data:
        return None
    kept, _ = reportable(data[REF], 10, 10)
    hosps = [h for h, _, _ in kept]
    if not hosps:
        return None
    data = {m: {h: v for h, v in d.items() if h in hosps} for m, d in data.items()}
    n_by = {h: len(data[REF][h][0]) for h in hosps}
    rng = np.random.default_rng(42)
    base = {m: macro_metric(data[m], metric=metric) for m in data}
    if base.get(REF) is None:
        return None
    idxs = [{h: rng.integers(0, n_by[h], n_by[h]) for h in hosps} for _ in range(n)]
    ref_vals = [macro_metric(data[REF], ix, metric) for ix in idxs]
    out: dict = {}
    for m in METHODS:
        if m not in data or base.get(m) is None:
            continue
        diffs = []
        for ix, r in zip(idxs, ref_vals):
            if r is None:
                continue
            a = macro_metric(data[m], ix, metric)
            if a is not None:
                diffs.append(a - r)
        if not diffs:
            continue
        d = np.asarray(diffs)
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[m] = (base[m] - base[REF], lo, hi)
    return out


def short(cell: str) -> str:
    return (cell.replace("results_v11full", "")
                .replace("_classaware_", " CA ")
                .replace("_random_", " RD ")
                .replace("hospital_mortality", "mort")
                .replace("icu_admission", "icu")
                .replace("readmission", "readm"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    # Which campaign to summarise (default: the reported grid).
    ap.add_argument("--glob", default="results_v11full*",
                    help="cell-directory glob (default: the reported Flower grid)")
    args = ap.parse_args()

    cells = sorted(c for c in glob.glob(args.glob)
                   if os.path.isdir(c) and glob.glob(os.path.join(c, "predictions_*.npz")))
    if not cells:
        sys.exit("no paired-testable cells found")

    for metric in ("auroc", "auprc", "ece"):
        better = "lower" if metric == "ece" else "higher"
        print("\n" + "=" * 96, flush=True)
        print(f"{metric.upper()}  ({better} is better)   delta vs population_only   "
              f"[+ = sig. better, ! = sig. WORSE]", flush=True)
        print("=" * 96, flush=True)
        hdr = "".join(
            f"{m.replace('_only', '').replace('standard_', '')[:12]:>13}"
            for m in METHODS)
        print(f"{'cell':<42}{hdr}", flush=True)
        for c in cells:
            v = verdict(c, metric, args.n)
            if not v:
                continue
            row = f"{short(c)[:41]:<42}"
            for m in METHODS:
                if m not in v:
                    row += f"{'-':>13}"
                    continue
                dd, lo, hi = v[m]
                sig = lo > 0 or hi < 0
                good = (dd < 0) if metric == "ece" else (dd > 0)
                mark = ("+" if good else "!") if sig else " "
                row += f"{f'{dd:+.4f}{mark}':>13}"
            print(row, flush=True)


if __name__ == "__main__":
    main()
