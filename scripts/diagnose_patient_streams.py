"""Why does patient-adapter selection reject every adapter?

Acceptance compares the personalised model's AUROC with the cohort model's on the
patient's own validation slice. If that slice holds one class, AUROC is None and the
adapter is rejected whatever its quality. This counts how often that happens, per
task and per ``min_samples``, without a GPU.

Labels go through the harness's real ``make_collate`` and ``label_fn``. Calling
``label_fn`` on a single ``dataset[i]`` is not the same: the collate wraps each
``expected`` into a list, and skipping it silently gives all zeros.

::

    python scripts/diagnose_patient_streams.py --task readmission --sweep 4,8,16,30,60
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from cocofl.evaluation.batching import make_eval_loader

from cocofl.constants import STATIC_DATA_FN
from cocofl.data.static_features import build_static_arrays
from cocofl.personalisation.loop import per_patient_subsets, split_indices_by_time
from cocofl.personalisation.run_ablation import TASKS, make_collate, make_label_fn
from cocofl.vocabulary import Vocabulary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="/home/jimin_hong/coco-fl/data/tokenized/mimic_train")
    ap.add_argument("--task", default="hospital_mortality")
    ap.add_argument("--min-samples", type=int, default=60)
    ap.add_argument("--max-patients", type=int, default=40)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.7)
    ap.add_argument("--sweep", default="")
    args = ap.parse_args()

    cohort = Path(args.cohort)
    Vocabulary.from_path(cohort)
    with open(cohort / STATIC_DATA_FN, "rb") as f:
        static = pickle.load(f)
    cat, num = build_static_arrays(static)
    collate_fn = make_collate(cat, num)
    ds = TASKS[args.task](cohort)
    label_fn = make_label_fn(args.task)

    def labels_for(idxs: list[int]) -> list[int]:
        """Labels for these dataset indices, via the real collate path."""
        out: list[int] = []
        # Length-bucketed loader: a plain DataLoader cannot stack windows of
        # different lengths.
        loader = make_eval_loader(ds, idxs, collate_fn, batch_size=32)
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                _, y = batch
                out.extend(int(v) for v in label_fn(y))
        return out

    def slices(idxs):
        early, late = split_indices_by_time(ds, idxs, train_frac=args.train_frac)
        if not early or not late:
            return None
        fit, val = split_indices_by_time(ds, early, train_frac=args.val_frac)
        if not fit or not val:
            return None
        return fit, val, late

    if args.sweep:
        print(f"task={args.task}   (labels via real collate path)")
        print("  min_samp  eligible  inspected  pt_with_pos  val_both_classes  eval_both")
        for ms in [int(x) for x in args.sweep.split(",")]:
            buckets = per_patient_subsets(ds, min_samples=ms)
            n = withpos = valboth = evalboth = 0
            for pid, idxs in buckets.items():
                if n >= args.max_patients:
                    break
                s = slices(idxs)
                if s is None:
                    continue
                fit, val, late = s
                n += 1
                yv, ye = labels_for(val), labels_for(late)
                withpos += (sum(labels_for(fit)) + sum(yv) + sum(ye)) > 0
                valboth += len(set(yv)) >= 2
                evalboth += len(set(ye)) >= 2
            print(f"  {ms:<9} {len(buckets):<9} {n:<10} {withpos:<12} {valboth:<17} {evalboth}")
        return

    buckets = per_patient_subsets(ds, min_samples=args.min_samples)
    print(f"task={args.task}  min_samples={args.min_samples}  eligible={len(buckets)}")
    n = valsingle = 0
    rows = []
    for pid, idxs in buckets.items():
        if n >= args.max_patients:
            break
        s = slices(idxs)
        if s is None:
            continue
        fit, val, late = s
        yf, yv, ye = labels_for(fit), labels_for(val), labels_for(late)
        n += 1
        valsingle += len(set(yv)) < 2
        rows.append((pid, len(idxs), len(yf), sum(yf), len(yv), sum(yv), len(ye), sum(ye)))
    print(f"  inspected={n}   VAL single-class={valsingle}/{n}  <-- drives accept/reject")
    print("\n  pid        n_win  fit(n/pos)   val(n/pos)   eval(n/pos)")
    for pid, nw, nf, pf, nv, pv, ne, pe in rows[:30]:
        print(f"  {pid:<10} {nw:<6} {nf}/{pf:<9} {nv}/{pv:<9} {ne}/{pe}")


if __name__ == "__main__":
    main()
