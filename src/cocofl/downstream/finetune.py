"""Fit a classification head on a frozen TFNFullModel and report AUROC.

Each batch gets the patient's static features (looked up by patient id in
static_data.pickle) and each window's real inter-event times, rebuilt by the collate
from the dataset.

Two probes, chosen with ``probe=``:

``balanced`` (default) Cached features and a class-weighted logistic head (see
:mod:`.probe`). Used for the reported population baselines: it keeps every positive,
weights the rare class up and selects on validation AUROC, which matters at ~2 %
positives. ``head`` The original SGD-trained MLP head, kept for comparison. It also
uses ``pos_weight``, class-stratified sampling and selection on validation AUROC.
"""
import copy
import json
import pickle
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..constants import STATIC_DATA_FN, horizon_microseconds
from ..data.static_features import build_static_arrays
from ..evaluation.batching import make_eval_loader
from ..models.classifier import TFNDownstreamClassifier
from ..models.backbone import TFNFullModel
from ..personalisation.run_ablation import POSITIVE_TOKENS, TASKS, make_collate
from ..vocabulary import Vocabulary
from .probe import (
    extract_features, fit_balanced_probe, fit_torch_head, score,
    stratified_subsample, window_labels,
)


def label_from_batch(label_dicts, pos_tokens, horizon_us=None):
    """Binary labels for a batch.

    With ``horizon_us`` set, the outcome token must also fall within that many
    microseconds of the prediction point; a later event counts as negative (e.g.
    readmission at 40 days under a 30-day horizon).
    """
    expected = label_dicts["expected"]
    if horizon_us is None:
        return [1 if e in pos_tokens else 0 for e in expected]
    times = label_dicts.get("true_token_time")
    if times is None:
        raise KeyError(
            "a label horizon was requested but the batch carries no "
            "'true_token_time'; the collate must forward it (see make_collate)."
        )
    return [
        1 if (e in pos_tokens and t is not None and 0 <= t <= horizon_us) else 0
        for e, t in zip(expected, times)
    ]


@torch.no_grad()
def val_auroc(classifier, loader, device, pos_tokens, max_iters=None, horizon_us=None):
    """Validation AUROC, the selection metric for the SGD head.

    Selecting on validation loss picks a near-constant predictor on imbalanced
    tasks, because unweighted BCE is lowest when predicting the base rate. AUROC
    does not reward that, and it is the metric that gets reported.
    """
    classifier.eval()
    preds, labels = [], []
    for i, batch in enumerate(loader):
        if batch is None:
            continue
        (x, elapsed, cat, num), y_dict = batch
        logits = classifier(
            input_ids=x.to(device), elapsed_times=elapsed.to(device),
            static_categorical=cat.to(device), static_numerical=num.to(device),
        )
        preds.extend(torch.sigmoid(logits).flatten().cpu().numpy().tolist())
        labels.extend(label_from_batch(y_dict, pos_tokens, horizon_us))
        if max_iters and i + 1 >= max_iters:
            break
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, preds))


def train_head(classifier, train_loader, val_loader, device, pos_tokens,
               max_iters, lr=1e-3, eval_every=100, patience=3, val_iters=200,
               horizon_us=None, pos_weight=None):
    """Fit the MLP head with class-weighted BCE, selecting on validation AUROC.

    ``pos_weight`` increases the loss on positives. At a 2 % base rate they would
    otherwise contribute ~2 % of the gradient and the head would just predict the
    majority class.
    """
    for p in classifier.backbone.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(classifier.classifier.parameters(), lr=lr)
    pw = None if pos_weight is None else torch.tensor(float(pos_weight), device=device)
    best_val, best_state, best_iter, stale = -1.0, copy.deepcopy(classifier.classifier.state_dict()), 0, 0
    pbar = tqdm(total=max_iters, desc="train")
    iters, stopped_early = 0, False
    while iters < max_iters and not stopped_early:
        for batch in train_loader:
            if batch is None:
                continue
            (x, elapsed, cat, num), y_dict = batch
            y = torch.tensor(label_from_batch(y_dict, pos_tokens, horizon_us), dtype=torch.float32, device=device)
            classifier.train()
            logits = classifier(
                input_ids=x.to(device), elapsed_times=elapsed.to(device),
                static_categorical=cat.to(device), static_numerical=num.to(device),
            )
            loss = F.binary_cross_entropy_with_logits(logits.flatten(), y, pos_weight=pw)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iters += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.3f}", best_val=f"{best_val:.3f}")
            if iters % eval_every == 0:
                vauc = val_auroc(classifier, val_loader, device, pos_tokens, max_iters=val_iters,
                                 horizon_us=horizon_us)
                if vauc is not None and vauc > best_val + 1e-4:
                    best_val, best_state, best_iter, stale = vauc, copy.deepcopy(classifier.classifier.state_dict()), iters, 0
                else:
                    stale += 1
                pbar.set_postfix(loss=f"{loss.item():.3f}",
                                 val="n/a" if vauc is None else f"{vauc:.3f}",
                                 best_val=f"{best_val:.3f}", stale=stale)
                if stale >= patience:
                    stopped_early = True
                    break
            if iters >= max_iters:
                break
    pbar.close()
    classifier.classifier.load_state_dict(best_state)
    print(f"  early stop: best val_auroc={best_val:.4f} @ iter {best_iter} (ran {iters}/{max_iters}, stopped_early={stopped_early})")
    return {"best_val_auroc": best_val, "best_iter": best_iter, "ran_iters": iters, "stopped_early": stopped_early}


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = Path(cfg.train_dir) if hasattr(cfg, "train_dir") and cfg.train_dir else None
    test_dir = Path(cfg.input_dir)
    model_path = Path(cfg.model_path)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_kind = str(getattr(cfg, "probe", "balanced")).lower()
    if probe_kind not in ("balanced", "head"):
        raise ValueError(f"probe must be 'balanced' or 'head', got {probe_kind!r}")
    # The balanced probe caches features instead of re-running the backbone every
    # epoch, so it can use many more training windows, which it needs to keep every
    # positive of a 2 % task.
    default_n_train = 20000 if probe_kind == "balanced" else 3000
    n_train = int(getattr(cfg, "n_train_samples", default_n_train))
    n_eval = int(getattr(cfg, "n_eval_samples", 3000))
    max_iters = int(getattr(cfg, "n_iters", 1500))
    val_frac = float(getattr(cfg, "val_frac", 0.1))
    eval_every = int(getattr(cfg, "eval_every", 100))
    patience = int(getattr(cfg, "patience", 3))
    batch_size = int(getattr(cfg, "batch_size", 16))
    # If set, also fit and save the torch classifier head per task, so WP4/WP5 start
    # from the real population model rather than a random head.
    head_dir = getattr(cfg, "save_head_dir", None)
    seed = int(getattr(cfg, "seed", 42))

    print(f"Loading TFNFullModel checkpoint from {model_path}")
    ck = torch.load(model_path, map_location=device, weights_only=False)
    cfg_ck = ck["model_config"]
    backbone = TFNFullModel(
        vocab_size=cfg_ck.vocab_size,
        n_embd=cfg_ck.n_embd,
        n_layer=cfg_ck.n_layer,
        n_positions=cfg_ck.n_positions,
        categorical_cardinalities=tuple(cfg_ck.categorical_cardinalities),
        numerical_dim=cfg_ck.numerical_dim,
        use_static=cfg_ck.use_static,
    )
    backbone.load_state_dict(ck["model"])
    backbone.to(device)
    Vocabulary.from_path(test_dir)  # validate vocab compatibility

    # Static features for the collate
    with open(test_dir / STATIC_DATA_FN, "rb") as f:
        test_static = pickle.load(f)
    test_cat, test_num = build_static_arrays(test_static)
    if train_dir is not None:
        with open(train_dir / STATIC_DATA_FN, "rb") as f:
            train_static = pickle.load(f)
        train_cat, train_num = build_static_arrays(train_static)
    else:
        train_cat, train_num = test_cat, test_num

    # The collates are built per task in the loop below, once the datasets exist,
    # because they need the dataset to rebuild each window's real time gaps.

    # Optional task filter: a comma-separated string or a list of task names.
    # Default: all tasks.
    task_filter = getattr(cfg, "tasks", None)
    if task_filter:
        if isinstance(task_filter, str):
            task_filter = [t.strip() for t in task_filter.split(",") if t.strip()]
        task_filter = set(task_filter)
        unknown = task_filter - set(TASKS)
        if unknown:
            raise ValueError(f"unknown task(s) {unknown}; valid: {list(TASKS)}")
        selected = {k: v for k, v in TASKS.items() if k in task_filter}
    else:
        selected = TASKS

    rng = np.random.default_rng(seed)
    results, rows = {}, []
    for task_name, dataset_cls in selected.items():
        print(f"\n=== {task_name} ({probe_kind} probe) ===")
        pos_tokens = POSITIVE_TOKENS[task_name]
        horizon_us = horizon_microseconds(task_name)
        if horizon_us is not None:
            print(f"  label horizon: {horizon_us / 8.64e10:.0f} days "
                  f"(outcome later than this counts as NEGATIVE)")

        train_base = dataset_cls(train_dir) if train_dir else None
        test_base = dataset_cls(test_dir)
        train_collate = make_collate(train_cat, train_num,
                                     dataset=train_base if train_base is not None else test_base)
        test_collate = make_collate(test_cat, test_num, dataset=test_base)

        # WP3: train the head only on a given patient list (e.g. the 40 % reserve),
        # so it never sees global-test patients when train_dir == input_dir.
        train_idx = list(range(len(train_base))) if train_base is not None else None
        train_restrict_path = getattr(cfg, "train_restrict_path", None)
        if train_base is not None and train_restrict_path:
            train_idx = filter_by_patients(train_base, train_restrict_path)
            print(f"  train_restrict: kept {len(train_idx)}/{len(train_base)} samples")

        # WP2 task 2: evaluate only on the 10 % global held-out test set.
        test_idx = list(range(len(test_base)))
        global_test_path = getattr(cfg, "global_test_path", None)
        if global_test_path:
            test_idx = filter_by_patients(test_base, global_test_path)
            print(f"  global_test filter: kept {len(test_idx)}/{len(test_base)} samples")

        # n_eval <= 0 means use the whole test set. A cap starves rare tasks: at ~2
        # % prevalence 3,000 windows hold only ~50 positives and mortality AUROC
        # jumps between runs. The full global test set has about 650.
        if 0 < n_eval < len(test_idx):
            test_idx = sorted(rng.choice(test_idx, size=n_eval, replace=False).tolist())
        print(f"  eval on {len(test_idx):,} test windows "
              f"({'FULL set' if n_eval <= 0 else f'cap={n_eval}'})")

        if probe_kind == "balanced":
            row = run_balanced_probe(
                backbone, train_base, train_idx, test_base, test_idx,
                train_collate, test_collate, pos_tokens, horizon_us,
                device, n_train, val_frac, batch_size, seed, rng,
                save_head_path=(Path(head_dir) / f"{task_name}.pt") if head_dir else None,
            )
        else:
            row = run_sgd_head(
                backbone, train_base, train_idx, test_base, test_idx,
                train_collate, test_collate, pos_tokens, horizon_us,
                device, n_train, val_frac, batch_size, max_iters, eval_every,
                patience, rng,
            )

        if row.get("auc") is None:
            print("  AUC undefined (test set has a single class) — skipped")
            continue
        results[task_name] = row["auc"]
        print(f"  AUC = {row['auc']:.4f}  AUPRC = {row['auprc']:.4f}  "
              f"(test prevalence {row['prevalence']:.3%}, {row['n_pos']:,} positives)")
        rows.append({"task": task_name, **row})
        pd.DataFrame(rows).to_csv(output_dir / "downstream_auc_backbone.csv", index=False)

    print("\n=== Final ===")
    for k, v in results.items():
        print(f"  {k:<25} AUC = {v:.4f}{'' if v >= 0.7 else '   <-- below 0.70'}")


def filter_by_patients(dataset, patient_list_path) -> list[int]:
    """Indices of samples in ``dataset`` whose prediction point belongs to a listed patient.

    ``start_indices[i]`` is the token index sample ``i`` ends on, which is what
    ``patient_id_at_idx`` is keyed by. A dataset without it is a bug: indexing by
    sample number would silently look up the wrong patient.
    """
    if not hasattr(dataset, "start_indices"):
        raise AttributeError(
            f"{type(dataset).__name__} exposes no 'start_indices', so its samples "
            "cannot be mapped back to patients for the split filter."
        )
    pids = {int(p) for p in json.loads(Path(patient_list_path).read_text())["patient_ids"]}
    starts = torch.as_tensor(dataset.start_indices, dtype=torch.long)
    # Look up all starts at once: on the sharded backend ``patient_id_at_idx``
    # rebuilds a lazy accessor on every access.
    at_start = dataset.patient_id_at_idx[starts]
    return [i for i, p in enumerate(at_start.reshape(-1).tolist()) if int(p) in pids]


def run_balanced_probe(backbone, train_base, train_idx, test_base, test_idx,
                       train_collate, test_collate, pos_tokens, horizon_us,
                       device, n_train, val_frac, batch_size, seed, rng,
                       save_head_path=None):
    """Cached-feature, class-weighted logistic probe. See :mod:`.probe`."""
    if train_base is None:
        raise ValueError("the balanced probe needs a train_dir to fit its head on")

    y_train_all = window_labels(train_base, pos_tokens, horizon_us)
    y_test_all = window_labels(test_base, pos_tokens, horizon_us)
    pool = y_train_all[np.asarray(train_idx, dtype=np.int64)]
    print(f"  train pool: {len(train_idx):,} windows, {int(pool.sum()):,} positive "
          f"({pool.mean():.3%})")

    sel = stratified_subsample(y_train_all, train_idx, n_train, rng)
    y_sel = y_train_all[np.asarray(sel, dtype=np.int64)]
    print(f"  stratified draw: {len(sel):,} windows, {int(y_sel.sum()):,} positive "
          f"({y_sel.mean():.3%})")
    if len(np.unique(y_sel)) < 2:
        raise ValueError(
            f"the training draw is single-class ({int(y_sel.sum())} positives of "
            f"{len(sel)}); nothing can be fit on it."
        )

    X = extract_features(backbone, train_base, sel, train_collate, device,
                         batch_size=batch_size, desc="train features")
    # Stratify the validation split too, so C is not chosen on a split with no
    # positives. This needs at least 2 of each class.
    stratify = y_sel if np.bincount(y_sel, minlength=2).min() >= 2 else None
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y_sel, test_size=val_frac, random_state=seed, stratify=stratify,
    )
    print(f"  fitting balanced logistic head on {len(y_tr):,} "
          f"(val {len(y_va):,}, {int(y_va.sum()):,} positive)")
    clf, best_c, val_auc = fit_balanced_probe(X_tr, y_tr, X_va, y_va, seed=seed)
    print(f"  selected C={best_c:g} (val AUROC "
          f"{'n/a' if val_auc is None else f'{val_auc:.4f}'})")

    X_te = extract_features(backbone, test_base, test_idx, test_collate, device,
                            batch_size=batch_size, desc="test features")
    y_te = y_test_all[np.asarray(test_idx, dtype=np.int64)]
    out = score(clf, X_te, y_te)
    out.update({"probe": "balanced", "C": best_c, "val_auroc": val_auc,
                "n_train": len(y_tr), "n_train_pos": int(y_tr.sum())})

    if save_head_path is not None:
        # Also fit the torch head that TFNDownstreamClassifier uses, so WP4/WP5
        # start from the real population model. The checkpoint holds only the
        # backbone, so without this the head would be random and population_only
        # would score at chance. The classifier uses last-token pooling only.
        n_embd = int(backbone.n_embd)
        # No second pass needed: "last_mean" pooling is concat(last, mean), so the
        # last-token features are the first n_embd columns of what is cached.
        assert X.shape[1] == 2 * n_embd, (
            f"expected concat(last, mean) features of width {2 * n_embd}, got {X.shape[1]}"
        )
        Xh_tr, Xh_va, yh_tr, yh_va = train_test_split(
            X[:, :n_embd], y_sel, test_size=val_frac, random_state=seed, stratify=stratify,
        )
        head_state, head_val_auc = fit_torch_head(
            Xh_tr, yh_tr, Xh_va, yh_va, hidden=n_embd, device=device, seed=seed,
        )
        head_test_auc = _score_torch_head(head_state, X_te[:, :n_embd], y_te, n_embd, device)
        save_head_path = Path(save_head_path)
        save_head_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"classifier": head_state, "n_embd": n_embd,
                    "val_auroc": head_val_auc, "test_auroc": head_test_auc},
                   save_head_path)
        print(f"  torch head -> {save_head_path}  "
              f"(val AUROC {head_val_auc:.4f}, test AUROC "
              f"{'n/a' if head_test_auc is None else f'{head_test_auc:.4f}'})")
        out["head_val_auroc"] = head_val_auc
        out["head_test_auroc"] = head_test_auc
    return out


def _score_torch_head(head_state, X, y, hidden, device):
    """Test AUROC of a saved head, so WP4's population baseline can be checked."""
    import torch.nn as nn

    if len(np.unique(y)) < 2:
        return None
    head = nn.Sequential(
        nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden // 2, 1),
    ).to(device)
    head.load_state_dict(head_state)
    head.eval()
    with torch.no_grad():
        p = torch.sigmoid(
            head(torch.as_tensor(X, dtype=torch.float32, device=device)).view(-1)
        ).cpu().numpy()
    return float(roc_auc_score(y, p))


def run_sgd_head(backbone, train_base, train_idx, test_base, test_idx,
                 train_collate, test_collate, pos_tokens, horizon_us,
                 device, n_train, val_frac, batch_size, max_iters, eval_every,
                 patience, rng):
    """The original SGD-trained MLP head, with the class-imbalance fixes."""
    classifier = TFNDownstreamClassifier(backbone, num_classes=1).to(device)
    es_info = {}

    if train_base is not None:
        y_train_all = window_labels(train_base, pos_tokens, horizon_us)
        sel = stratified_subsample(y_train_all, train_idx, n_train, rng)
        y_sel = y_train_all[np.asarray(sel, dtype=np.int64)]
        n_pos = int(y_sel.sum())
        # Heavier loss on the rare class; without it the head just learns the base
        # rate.
        pos_weight = (len(y_sel) - n_pos) / max(n_pos, 1)
        print(f"  stratified draw: {len(sel):,} windows, {n_pos:,} positive "
              f"({y_sel.mean():.3%}), pos_weight={pos_weight:.1f}")

        n_val = max(1, int(len(sel) * val_frac))
        perm = rng.permutation(len(sel))
        val_sel = [sel[i] for i in perm[:n_val]]
        fit_sel = [sel[i] for i in perm[n_val:]]
        train_loader = DataLoader(
            Subset(train_base, fit_sel), batch_size=1, shuffle=True, collate_fn=train_collate,
        )
        val_loader = make_eval_loader(train_base, val_sel, train_collate, batch_size=batch_size)
        print(f"  train={len(fit_sel)} val={len(val_sel)}")
        print(f"Training head (max_iters={max_iters}, eval_every={eval_every}, patience={patience})")
        es_info = train_head(
            classifier, train_loader, val_loader, device, pos_tokens,
            horizon_us=horizon_us, pos_weight=pos_weight,
            max_iters=max_iters, eval_every=eval_every, patience=patience,
        )

    print("Evaluating")
    test_loader = make_eval_loader(test_base, test_idx, test_collate, batch_size=batch_size)
    preds, labels = [], []
    with torch.no_grad():
        classifier.eval()
        for batch in tqdm(test_loader, desc="eval"):
            if batch is None:
                continue
            (x, elapsed, cat, num), y_dict = batch
            logits = classifier(
                input_ids=x.to(device), elapsed_times=elapsed.to(device),
                static_categorical=cat.to(device), static_numerical=num.to(device),
            )
            preds.extend(torch.sigmoid(logits).flatten().cpu().numpy().tolist())
            labels.extend(label_from_batch(y_dict, pos_tokens, horizon_us))
    y_te, p = np.asarray(labels), np.asarray(preds)
    if len(np.unique(y_te)) < 2:
        return {"probe": "head", "auc": None, **es_info}
    return {
        "probe": "head",
        "auc": float(roc_auc_score(y_te, p)),
        "auprc": float(average_precision_score(y_te, p)),
        "n": int(len(y_te)),
        "n_pos": int(y_te.sum()),
        "prevalence": float(y_te.mean()),
        **es_info,
    }


if __name__ == "__main__":
    main()
