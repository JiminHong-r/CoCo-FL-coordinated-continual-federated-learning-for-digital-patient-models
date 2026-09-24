"""Do centres disagree about how features map to risk? (concept shift in feature space)

``heterogeneity_analysis.py`` tests a two-parameter reshaping of the population
model's final score. A cohort adapter can do more: it transforms the whole final
hidden state before a frozen head. This test works at that resolution.

No data is pooled across centres. Each head is fitted on one centre's frozen
features and evaluated on every centre's test partition::

              tested on h0   tested on h1   tested on h2
    head h0       AUROC          AUROC          AUROC
    head h1       ...
    head h2       ...

Compare the diagonal with the off-diagonal. If a centre's own head beats the best
head trained elsewhere, the centres disagree about how features map to risk, which
personalisation could exploit. If not, they share the same conditional law at the
adapter's resolution. The only thing crossing a centre boundary is a fitted head
being evaluated, which is a measurement, not training.

Every head is fitted on the same number of windows, with the same stratification.
Centres differ hugely in size (at alpha=0.1, ~28,000 windows in one and ~500 in
another), and a head fitted on more data would win for that reason alone. (An
earlier version compared against a head fitted on features pooled across centres;
pooling is not allowed in a federated setting, so that comparison was dropped.)

::

    python scripts/cross_centre_transfer.py \\
        --test-dir   data/tokenized/mimic_train \\
        --model-path models/population_reserve_nonotes_full/best_model.pt \\
        --splits-dir data/federated_splits_v2_classaware \\
        --task hospital_mortality --alpha 0.5
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
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
                    help="windows per centre head, matched across centres")
    ap.add_argument("--n-test", type=int, default=12000,
                    help="cap on test windows per centre, for runtime")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = Path(args.test_dir)
    rng = np.random.default_rng(args.seed)

    Vocabulary.from_path(test_dir)
    with open(test_dir / STATIC_DATA_FN, "rb") as f:
        static = pickle.load(f)
    cat, num = build_static_arrays(static)

    ds = TASKS[args.task](test_dir)
    # Build the collate with the dataset so the T-LSTM gets real time gaps.
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

    train_pool = {h: list(tr_split.indices_per_hospital[h]) for h in hosps}
    budget = min(args.n_train, min(len(v) for v in train_pool.values()))
    print(f"matched training budget: {budget:,} windows per centre")

    feats_tr, ys_tr, feats_te, ys_te = {}, {}, {}, {}
    for h in hosps:
        sel = stratified_subsample(y_all, train_pool[h], budget, rng)
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
        ys_te[h] = y_all[np.asarray(te_idx, dtype=np.int64)]
        feats_te[h] = extract_features(backbone, ds, te_idx, collate, device,
                                       batch_size=args.batch_size,
                                       desc=f"h{h} test feats")
        print(f"  h{h}: train {len(y_sel):,} ({int(y_sel.sum())} pos)  "
              f"test {len(ys_te[h]):,} ({int(ys_te[h].sum())} pos)")

    usable = [h for h in hosps
              if h in feats_tr and len(np.unique(ys_te.get(h, []))) > 1]
    if len(usable) < 2:
        sys.exit("fewer than two centres are usable; a transfer matrix is undefined")

    heads = {}
    for h in usable:
        Xtr, Xva, ytr, yva = train_test_split(
            feats_tr[h], ys_tr[h], test_size=0.2, random_state=args.seed,
            stratify=ys_tr[h] if np.bincount(ys_tr[h], minlength=2).min() >= 2 else None,
        )
        heads[h], best_c, _ = fit_balanced_probe(Xtr, ytr, Xva, yva, seed=args.seed)
        print(f"  head h{h}: C={best_c:g}  (fitted on h{h} only)")

    header = "".join(f"{'h' + str(h):>12}" for h in usable)
    print(f"\n  {'head \\ tested on':<20}{header}")
    rows = {}
    for hh in usable:
        row = {ht: _auc(heads[hh], feats_te[ht], ys_te[ht]) for ht in usable}
        rows[hh] = row
        line = "".join("  n/a" if row[ht] is None else f"{row[ht]:>12.4f}"
                       for ht in usable)
        print(f"  {'h' + str(hh):<20}{line}")

    print("\n  own-centre advantage (its own head minus the best head trained elsewhere)")
    diffs = []
    for h in usable:
        own = rows[h][h]
        others = [rows[o][h] for o in usable if o != h and rows[o][h] is not None]
        if own is None or not others:
            continue
        d = own - max(others)
        diffs.append(d)
        print(f"    h{h}: own {own:.4f}  best other {max(others):.4f}  -> {d:+.4f}")

    if diffs:
        print(f"\n  mean own-vs-other {np.mean(diffs):+.4f}")
    print("\n  Positive and consistent => centres disagree about how features map to")
    print("  risk (concept shift), and personalisation has a target the score-space")
    print("  test could not see.")
    print("  Around zero => the conditional law is shared at full representational")
    print("  resolution, and the score-space null of RESULTS.md 1d extends to")
    print("  feature space.")


if __name__ == "__main__":
    main()
