"""Frozen-backbone probe for the WP3 population baselines, built for class imbalance.

The original probe (``finetune.train_head``) fitted the head with unweighted BCE at
batch 1 on a 3,000-window random draw and kept the checkpoint with the lowest
validation loss. At ~2 % positives (mortality) that fails: the draw holds about 60
positives, each step sees a single example, and unweighted BCE is lowest when the
head just predicts the base rate. It gave a mortality AUROC of 0.589.

This probe instead:

1. Labels every candidate window up front from ``start_indices`` and
``outcome_indices``, without building any windows, so the class balance is known
before fitting. 2. Draws a class-stratified training sample: every positive, then
negatives to fill the budget. 3. Runs the frozen backbone once over the sampled
windows and caches the features (re-running a frozen backbone every epoch gains
nothing). 4. Fits class-weighted logistic regressions on the cached features and
picks the regularisation strength by validation AUROC, the reported metric.

The features are the last hidden state (what ``TFNDownstreamClassifier`` uses)
concatenated with the mean over the window.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..evaluation.batching import LengthBucketBatchSampler, sequence_lengths

# Regularisation grid for the balanced logistic head, chosen by validation AUROC.
# The small values matter: the features are 1024-dimensional and mortality has only
# a few thousand positives.
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def window_labels(dataset, pos_tokens, horizon_us: float | None = None) -> np.ndarray:
    """Binary label for every sample of ``dataset``, without building windows.

    Gives the same result as running ``finetune.label_from_batch`` over the dataset:
    a sample is positive when its outcome token is in ``pos_tokens`` and, if
    ``horizon_us`` is set, falls within that many microseconds of the prediction
    point. Both come from the dataset's index tensors, so this is one vectorised
    pass.
    """
    outcomes = torch.as_tensor(dataset.outcome_indices)
    # Token positions holding a positive outcome token; ``isin`` against these is
    # the vectorised "decode(tokens[outcome_idx]) in pos_tokens".
    pos_positions = dataset._get_indices_of_stokens(sorted(pos_tokens))
    y = torch.isin(outcomes, pos_positions).numpy().astype(np.int8)

    if horizon_us is not None:
        # Only rows with a positive token can become negative, so the per-index time
        # lookups (not vectorisable on sharded storage) run on that small subset.
        starts = torch.as_tensor(dataset.start_indices)
        times = dataset.times
        for i in np.flatnonzero(y):
            dt = int(times[int(outcomes[i])]) - int(times[int(starts[i])])
            if not 0 <= dt <= horizon_us:
                y[i] = 0
    return y


def stratified_subsample(
    y: np.ndarray,
    candidates,
    n_total: int,
    rng: np.random.Generator,
    max_neg_per_pos: float | None = None,
) -> list[int]:
    """Pick at most ``n_total`` of ``candidates``, keeping as many positives as possible.

    Positives are taken first, up to half the budget (so a common outcome cannot end
    up all-positive), and negatives fill the rest. A random draw at a 2 % base rate
    would hold only ~60 positives in 3,000 windows.
    """
    candidates = np.asarray(list(candidates), dtype=np.int64)
    labels = y[candidates]
    pos, neg = candidates[labels == 1], candidates[labels == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    if n_total <= 0 or n_total >= len(candidates):
        keep_pos, keep_neg = pos, neg
    else:
        n_pos = min(len(pos), max(1, n_total // 2))
        keep_pos = pos[:n_pos]
        keep_neg = neg[: n_total - n_pos]

    if max_neg_per_pos is not None and len(keep_pos):
        keep_neg = keep_neg[: int(max_neg_per_pos * len(keep_pos))]

    out = np.concatenate([keep_pos, keep_neg])
    rng.shuffle(out)
    return out.tolist()


def fit_torch_head(X_tr, y_tr, X_va, y_va, hidden: int, device,
                   epochs: int = 40, lr: float = 1e-3, batch_size: int = 256,
                   seed: int = 42):
    """Fit the MLP head that ``TFNDownstreamClassifier`` uses, on cached features.

    The population checkpoint holds only the backbone, so
    ``TFNDownstreamClassifier`` would otherwise start with a random head and
    ``population_only`` would score at chance. Fitting it once here, on the reserve,
    gives every downstream level the real population model.

    Features must use last-token pooling (``pooling="last"``), because that is what
    ``TFNDownstreamClassifier.forward`` feeds its head.

    Returns ``(state_dict, val_auroc)``; the state dict loads straight into
    ``TFNDownstreamClassifier.classifier``.
    """
    torch.manual_seed(seed)
    head = nn.Sequential(
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden // 2, 1),
    ).to(device)

    Xt = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(np.asarray(y_tr), dtype=torch.float32, device=device)
    Xv = torch.as_tensor(X_va, dtype=torch.float32, device=device)
    # Class-weighted, like the logistic probe: at ~2 % prevalence an unweighted head
    # learns the base rate.
    n_pos = float(yt.sum().item())
    pos_weight = torch.tensor(
        (len(yt) - n_pos) / max(n_pos, 1.0), device=device, dtype=torch.float32,
    )
    opt = torch.optim.AdamW(head.parameters(), lr=lr)

    best_state = copy.deepcopy(head.state_dict())
    best_auc = -1.0
    n = len(Xt)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = F.binary_cross_entropy_with_logits(
                head(Xt[idx]).view(-1), yt[idx], pos_weight=pos_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            p = torch.sigmoid(head(Xv).view(-1)).cpu().numpy()
        if len(np.unique(y_va)) > 1:
            auc = roc_auc_score(y_va, p)
            # Select on validation AUROC, never on loss.
            if auc > best_auc:
                best_auc, best_state = auc, copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    return {k: v.detach().cpu() for k, v in head.state_dict().items()}, best_auc


@torch.no_grad()
def extract_features(
    backbone,
    dataset,
    indices,
    collate_fn,
    device,
    batch_size: int = 16,
    desc: str = "features",
    pooling: str = "last_mean",
) -> np.ndarray:
    """Pooled frozen-backbone features for each of ``indices``.

    Row ``k`` belongs to ``indices[k]``. Batches are grouped by exact sequence
    length (see ``evaluation.batching``), so there is no padding and every row
    matches a ``batch_size=1`` pass.
    """
    indices = list(indices)
    lengths = sequence_lengths(dataset, indices) if batch_size > 1 else None
    if lengths is None:
        batches = [[p] for p in range(len(indices))]
    else:
        batches = list(LengthBucketBatchSampler(lengths, batch_size))

    backbone.eval()
    out = None
    for positions in tqdm(batches, desc=desc):
        items = [dataset[indices[p]] for p in positions]
        batch = collate_fn(items)
        if batch is None:
            raise RuntimeError(
                f"collate dropped every window of a {len(positions)}-sample batch; "
                "features would be silently zero for those rows."
            )
        (x, elapsed, cat, num), _ = batch
        if len(x) != len(positions):
            raise RuntimeError(
                f"collate returned {len(x)} rows for {len(positions)} windows; "
                "the feature matrix would be misaligned with its labels."
            )
        hidden = backbone(
            input_ids=x.to(device), elapsed_times=elapsed.to(device),
            static_categorical=cat.to(device), static_numerical=num.to(device),
            labels=None,
        ).hidden_states
        # "last" is what TFNDownstreamClassifier.forward feeds its head, so a head
        # fitted on it loads straight into that model. "last_mean" adds a
        # whole-window summary, which helps the linear probe but has no slot in the
        # classifier.
        if pooling == "last":
            pooled = hidden[:, -1, :]
        elif pooling == "last_mean":
            pooled = torch.cat([hidden[:, -1, :], hidden.mean(dim=1)], dim=-1)
        else:
            raise ValueError(f"unknown pooling {pooling!r}")
        feats = pooled.float().cpu().numpy()
        if out is None:
            out = np.zeros((len(indices), feats.shape[1]), dtype=np.float32)
        out[positions] = feats
    if out is None:
        raise RuntimeError("feature extraction produced no batches")
    return out


def fit_balanced_probe(X_tr, y_tr, X_va, y_va, seed: int = 42, c_grid=C_GRID):
    """Class-weighted logistic head, with the regularisation picked on validation AUROC.

    ``class_weight="balanced"`` weights each class by the inverse of its frequency,
    so the rare class counts as much in total as the common one.
    """
    best, best_key = None, -float("inf")
    for c in c_grid:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c, class_weight="balanced", max_iter=5000, random_state=seed,
            ),
        )
        clf.fit(X_tr, y_tr)
        # A single-class validation split has no AUROC; still fit, so the caller
        # gets a head, but do not let it drive selection.
        auc = (
            roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
            if len(np.unique(y_va)) > 1
            else None
        )
        print(f"    C={c:<8g} val AUROC={'n/a' if auc is None else f'{auc:.4f}'}")
        key = -float("inf") if auc is None else auc
        if best is None or key > best_key:
            best, best_key = (clf, c, auc), key
    return best


def score(clf, X, y) -> dict[str, float | None]:
    """Test AUROC and AUPRC, plus the prevalence they were measured at."""
    p = clf.predict_proba(X)[:, 1]
    if len(np.unique(y)) < 2:
        return {"auc": None, "auprc": None, "n": len(y), "n_pos": int(y.sum())}
    return {
        "auc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalence": float(y.mean()),
    }
