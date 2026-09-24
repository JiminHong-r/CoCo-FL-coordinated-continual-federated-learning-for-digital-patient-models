"""Class-imbalance-aware frozen-backbone probe for the WP3 population baselines.

Why this exists
---------------
The original probe (``finetune.train_head``) fit the MLP head with *unweighted*
BCE, at ``batch_size=1``, on a 3,000-window uniform draw, and early-stopped on
validation **loss**. On a low-prevalence task that combination is degenerate:

* ``hospital_mortality`` runs at ~2 % positives, so a 3,000-window uniform draw
  carries ~60 positives — and only ~54 of them ever reach the optimiser, since
  a 1,500-iteration budget at batch 1 sees 1,500 windows total.
* Every gradient step sees exactly one example, so the update is the label
  itself, not a class-balanced estimate of the gradient.
* Unweighted BCE is minimised by predicting the base rate for everybody, and
  because selection is on that same loss, the checkpoint that gets *kept* is
  the one closest to the constant predictor. The ranking it produces is close
  to arbitrary, which is exactly the 0.589 mortality AUROC that came out.

What this does instead
----------------------
1. Labels every candidate window up front, straight from ``start_indices`` /
   ``outcome_indices``, without constructing a single window — so the class
   balance is known *before* anything is fit, and can be reported.
2. Draws a **class-stratified** training sample: keep every positive, then fill
   the budget with negatives. The head no longer depends on a lucky draw.
3. Runs the frozen backbone **once** over the sampled train windows and the
   test windows, caching the pooled representation. The backbone is frozen, so
   re-running it every epoch (as the SGD path does) buys nothing.
4. Fits **class-weighted** logistic regressions on the cached features and
   selects the regularisation strength by validation **AUROC** — the metric
   that is actually reported, rather than a loss that rewards the base rate.

Pooling concatenates the last hidden state (what ``TFNDownstreamClassifier``
uses) with the mean over the timeline, which gives the linear head a summary of
the whole window rather than only its final step.
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

# Regularisation grid for the balanced logistic head. Selected on validation
# AUROC; the low end matters because the pooled features are 1024-dimensional
# and the positive count on mortality is in the low thousands.
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def window_labels(dataset, pos_tokens, horizon_us: float | None = None) -> np.ndarray:
    """Binary label for every sample of ``dataset``, without building windows.

    Equivalent to running ``finetune.label_from_batch`` over the whole dataset:
    a sample is positive when its outcome token is one of ``pos_tokens`` and —
    when ``horizon_us`` is set — the outcome falls within that many
    microseconds of the prediction point. Both facts are already available from
    the dataset's index tensors, so this costs one vectorised pass instead of
    ``len(dataset)`` window constructions.
    """
    outcomes = torch.as_tensor(dataset.outcome_indices)
    # Token positions at which a positive outcome token sits. ``isin`` against
    # these is the vectorised form of "decode(tokens[outcome_idx]) in pos_tokens".
    pos_positions = dataset._get_indices_of_stokens(sorted(pos_tokens))
    y = torch.isin(outcomes, pos_positions).numpy().astype(np.int8)

    if horizon_us is not None:
        # Only the token-positive rows can be demoted, so the per-index time
        # lookups (which are not vectorisable over sharded storage) run over a
        # small subset.
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
    """Pick <= ``n_total`` of ``candidates``, keeping as many positives as possible.

    A uniform draw is what starves the low-prevalence tasks: at a 2 % base rate
    a 3,000-window draw holds ~60 positives. Here positives are taken first (up
    to half the budget, so a high-prevalence task cannot end up all-positive)
    and negatives fill the remainder.
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

    Why this exists: the population checkpoint contains only the backbone and
    its language-model head, so ``TFNDownstreamClassifier(backbone)`` starts
    with a **randomly initialised** classifier. WP4/WP5 therefore began from
    backbone + noise rather than from the WP3 population model, which made
    ``population_only`` a chance-level baseline and ``delta_cohort`` a measure
    of "was a head trained at all". Fitting it here once on the reserve gives
    every downstream level the population model the project description assumes.

    Features must be LAST-token pooled (``pooling="last"``) because that is what
    ``TFNDownstreamClassifier.forward`` feeds its head.

    Returns ``(state_dict, val_auroc)`` where the state dict drops straight into
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
    # Class-weighted, for the same reason the logistic probe is: at ~2 %
    # prevalence an unweighted head learns the base rate.
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
            # Select on validation AUROC, never on loss — same trap as the
            # original probe, which kept the checkpoint closest to the prior.
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
    """Pooled frozen-backbone representation for each of ``indices``.

    Row ``k`` of the result corresponds to ``indices[k]``. Batches are bucketed
    by exact sequence length (see ``evaluation.batching``) so no padding ever
    enters the T-LSTM recurrence and each row is identical to what a
    ``batch_size=1`` pass would produce.
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
        # ``last`` matches what TFNDownstreamClassifier.forward feeds its head,
        # so a head fitted on these features drops straight into that model.
        # ``last_mean`` adds a whole-window summary, which helps the linear
        # probe but has no slot in the existing classifier.
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
    """Class-weighted logistic head, regularisation picked on validation AUROC.

    ``class_weight="balanced"`` scales each class's contribution by the inverse
    of its frequency, so the minority class carries the same total weight as
    the majority one — the "higher penalty on the rare class" fix, applied to
    the loss the head is actually fit with.
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
        # A single-class validation split leaves AUROC undefined; keep fitting
        # so the caller still gets a head, but never let it drive selection.
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
    """Test-set AUROC/AUPRC plus the prevalence they were measured at."""
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
