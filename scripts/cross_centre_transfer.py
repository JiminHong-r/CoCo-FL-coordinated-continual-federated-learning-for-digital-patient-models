"""Does centre-specific structure exist in FEATURE space, not just score space?

WHY THIS EXISTS
---------------
`heterogeneity_analysis.py` finds no exploitable per-centre structure, but it tests
a two-parameter family — `sigma(a * logit(score) + b)` — i.e. a monotone reshaping
of the population model's final SCORE. A cohort adapter is more expressive than
that: it transforms the whole final hidden state before a frozen head, so it can
rotate the representation and change what the head's fixed direction reads. The
score-space null therefore does not, on its own, license the claim "there is no
centre-specific signal".

This script closes that gap at full representational resolution. Fit a linear head
on frozen features PER CENTRE, then score every head on every centre's test data:

              tested on h0   tested on h1   tested on h2
    head h0       AUROC          AUROC          AUROC
    head h1       ...
    head h2       ...
    pooled        ...

If the DIAGONAL beats the off-diagonal, centres disagree about how features map to
risk — concept shift, and a genuine target for personalisation. If the diagonal is
level with the off-diagonal, the conditional law is shared at the resolution the
adapter operates on, and the null is established far more strongly than before.

SIZE MATCHING IS THE POINT, NOT A DETAIL
----------------------------------------
Centres differ enormously in size (α=0.1 puts ~28,000 windows in one and ~500 in
another). A head fitted on more data wins for reasons that have nothing to do with
concept shift, so an unmatched diagonal advantage would be unreadable. By default
every centre's head is fitted on the SAME number of windows, drawn with the same
stratification — so any diagonal advantage that survives is about the *mapping*,
not the sample size. `--no-match-n` disables it to show the contrast.

The pooled head answers the complementary question the supervisor raised: what does
partitioning cost when the estimator is a simple linear probe rather than an
adapter? It is fitted on the same matched budget, drawn across all centres.

    python scripts/cross_centre_transfer.py \
        --test-dir  data/tokenized/mimic_train \
        --model-path models/population_reserve_nonotes_full/best_model.pt \
        --splits-dir data/federated_splits_v2_classaware \
        --task hospital_mortality --alpha 0.5
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from cocofl.constants import STATIC_DATA_FN, horizon_microseconds
from cocofl.data.static_features import build_static_arrays
from cocofl.downstream.probe import (
    extract_features,
    fit_balanced_probe,
    stratified_subsample,
    window_labels,
)
from cocofl.federated import as_hospital_split, load_split
from cocofl.hierarchical.run_hierarchical import build_population_classifier
from cocofl.personalisation.run_ablation import POSITIVE_TOKENS, TASKS, make_collate
from cocofl.vocabulary import Vocabulary
from sklearn.metrics import roc_auc_score


def _auc(clf, X, y) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--n-train", type=int, default=8000,
                    help="windows per centre head (matched across centres)")
    ap.add_argument("--n-test", type=int, default=12000,
                    help="cap on test windows per centre, for runtime")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-match-n", action="store_true",
                    help="fit each head on all its data (confounds size with shift)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = Path(args.test_dir)
    rng = np.random.default_rng(args.seed)

    Vocabulary.from_path(test_dir)
    with open(test_dir / STATIC_DATA_FN, "rb") as f:
        static = pickle.load(f)
    cat, num = build_static_arrays(static)

    ds = TASKS[args.task](test_dir)
    # Rebuild collate WITH the dataset so the T-LSTM receives real inter-event
    # times; a constant would run the backbone outside its training distribution.
    collate = make_collate(cat, num, dataset=ds)
    y_all = window_labels(ds, POSITIVE_TOKENS[args.task],
                          horizon_microseconds(args.task))

    backbone = build_population_classifier(Path(args.model_path), device).backbone
    backbone.eval()

    split = load_split(args.splits_dir, args.alpha)
    tr_split = as_hospital_split(split, partition="train")
    te_split = as_hospital_split(split, partition="test")
    for s in (tr_split, te_split):
        s.bind_to_dataset(ds)

    hosps = sorted(h for h, idx in tr_split.indices_per_hospital.items() if idx)
    print(f"task={args.task}  alpha={args.alpha}  centres={hosps}  device={device}")

    # ---- training budget, matched across centres by default ----------------
    train_pool = {h: list(tr_split.indices_per_hospital[h]) for h in hosps}
    smallest = min(len(v) for v in train_pool.values())
    budget = min(args.n_train, smallest) if not args.no_match_n else None
    if budget is not None:
        print(f"matched training budget: {budget:,} windows per centre "
              f"(smallest centre has {smallest:,})")
    else:
        print("UNMATCHED budget: each head uses all its centre's data — a diagonal "
              "advantage here may be sample size, not concept shift")

    feats_tr, ys_tr, feats_te, ys_te = {}, {}, {}, {}
    for h in hosps:
        n = budget if budget is not None else len(train_pool[h])
        sel = stratified_subsample(y_all, train_pool[h], n, rng)
        y_sel = y_all[np.asarray(sel, dtype=np.int64)]
        if len(np.unique(y_sel)) < 2:
            print(f"  h{h}: training draw is single-class — cannot fit; skipping")
            continue
        feats_tr[h] = extract_features(backbone, ds, sel, collate, device,
                                       batch_size=args.batch_size,
                                       desc=f"h{h} train feats")
        ys_tr[h] = y_sel

        te_idx = list(te_split.indices_per_hospital.get(h, []))
        if len(te_idx) > args.n_test:
            te_idx = list(rng.choice(te_idx, args.n_test, replace=False))
        y_te = y_all[np.asarray(te_idx, dtype=np.int64)]
        feats_te[h] = extract_features(backbone, ds, te_idx, collate, device,
                                       batch_size=args.batch_size,
                                       desc=f"h{h} test feats")
        ys_te[h] = y_te
        print(f"  h{h}: train {len(y_sel):,} ({int(y_sel.sum())} pos)  "
              f"test {len(y_te):,} ({int(y_te.sum())} pos)")

    usable = [h for h in hosps if h in feats_tr and len(np.unique(ys_te.get(h, []))) > 1]
    if len(usable) < 2:
        sys.exit("fewer than two centres are usable; a transfer matrix is undefined")

    # ---- fit one head per centre, plus a pooled head ------------------------
    heads = {}
    for h in usable:
        Xtr, Xva, ytr, yva = train_test_split(
            feats_tr[h], ys_tr[h], test_size=0.2, random_state=args.seed,
            stratify=ys_tr[h] if np.bincount(ys_tr[h], minlength=2).min() >= 2 else None,
        )
        clf, best_c, _ = fit_balanced_probe(Xtr, ytr, Xva, yva, seed=args.seed)
        heads[h] = clf
        print(f"  head h{h}: C={best_c:g}")

    # Pooled head on the SAME total budget, so "pooled" is not simply "more data".
    Xp = np.concatenate([feats_tr[h] for h in usable])
    yp = np.concatenate([ys_tr[h] for h in usable])
    keep = rng.choice(len(yp), min(len(yp), budget or len(yp)), replace=False)
    Xp, yp = Xp[keep], yp[keep]
    Xtr, Xva, ytr, yva = train_test_split(
        Xp, yp, test_size=0.2, random_state=args.seed,
        stratify=yp if np.bincount(yp, minlength=2).min() >= 2 else None)
    pooled_clf, pooled_c, _ = fit_balanced_probe(Xtr, ytr, Xva, yva, seed=args.seed)
    heads["pooled"] = pooled_clf
    print(f"  head pooled: C={pooled_c:g} on {len(yp):,} windows "
          f"(same budget as one centre)")

    # ---- the transfer matrix ------------------------------------------------
    print(f"\n  {'head \\ tested on':<20}" + "".join(f"{'h' + str(h):>12}" for h in usable))
    rows = {}
    for hh in list(usable) + ["pooled"]:
        row = {}
        line = f"  {('h' + str(hh)) if hh != 'pooled' else 'pooled':<20}"
        for ht in usable:
            a = _auc(heads[hh], feats_te[ht], ys_te[ht])
            row[ht] = a
            line += f"{'  n/a' if a is None else f'{a:>12.4f}'}"
        rows[hh] = row
        print(line)

    # ---- the read-out -------------------------------------------------------
    print("\n  own-centre advantage (its own head minus the best OTHER centre's head)")
    diffs = []
    for h in usable:
        own = rows[h][h]
        others = [rows[o][h] for o in usable if o != h and rows[o][h] is not None]
        if own is None or not others:
            continue
        d = own - max(others)
        diffs.append(d)
        print(f"    h{h}: own {own:.4f}  best other {max(others):.4f}  -> {d:+.4f}")
    print("\n  own-centre advantage over the POOLED head (same training budget)")
    pdiffs = []
    for h in usable:
        own, pool = rows[h][h], rows["pooled"][h]
        if own is None or pool is None:
            continue
        pdiffs.append(own - pool)
        print(f"    h{h}: own {own:.4f}  pooled {pool:.4f}  -> {own - pool:+.4f}")

    if diffs:
        print(f"\n  mean own-vs-other  {np.mean(diffs):+.4f}")
    if pdiffs:
        print(f"  mean own-vs-pooled {np.mean(pdiffs):+.4f}")
    print("\n  Positive and consistent => concept shift exists in feature space and")
    print("  personalisation has a target the score-space test could not see.")
    print("  Around zero => the conditional law is shared at the resolution the")
    print("  cohort adapter operates on, and the null is established at full")
    print("  representational resolution rather than only in score space.")


if __name__ == "__main__":
    main()
