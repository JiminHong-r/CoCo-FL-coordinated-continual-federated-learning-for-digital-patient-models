"""Evaluation for TFN-based personalisation.

Mirrors ``personalize.evaluate`` but works on the TFNDownstreamClassifier with
TFN-format batches: ``((x, elapsed, cat, num), y_dict)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ..models.classifier import TFNDownstreamClassifier


@dataclass
class AucResult:
    auc: float | None
    n: int
    n_pos: int


@dataclass
class MetricsResult:
    """All-in-one predictive metrics on a single (model, eval_loader) pair."""
    auroc: float | None
    auprc: float | None
    ece: float | None
    n: int
    n_pos: int
    # The per-window predictions these metrics were computed from. Every method
    # is scored on IDENTICAL windows, so their differences are PAIRED and far
    # better determined than independent per-method confidence intervals suggest
    # — but only if the predictions survive to be tested. A finished cell cannot
    # regenerate them, so they are carried here rather than recomputed.
    preds: list[float] | None = None
    labels: list[int] | None = None


@torch.no_grad()
def predict(
    classifier: TFNDownstreamClassifier,
    loader: DataLoader,
    label_fn: Callable[[dict], list[int]],
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[list[float], list[int]]:
    classifier.eval()
    preds: list[float] = []
    labels: list[int] = []
    for i, batch in enumerate(loader):
        if batch is None:
            continue
        x_tuple, y_dict = batch
        x, elapsed, cat, num = (t.to(device, non_blocking=True) for t in x_tuple[:4])
        logits = classifier(
            input_ids=x, elapsed_times=elapsed,
            static_categorical=cat, static_numerical=num,
        )
        probs = torch.sigmoid(logits.view(-1))
        preds.extend(probs.cpu().tolist())
        labels.extend(label_fn(y_dict))
        if max_batches is not None and i + 1 >= max_batches:
            break
    return preds, labels


def auc_from_preds(preds: list[float], labels: list[int]) -> AucResult:
    n = len(labels)
    n_pos = sum(labels)
    if len(set(labels)) < 2:
        return AucResult(auc=None, n=n, n_pos=n_pos)
    return AucResult(auc=float(roc_auc_score(labels, preds)), n=n, n_pos=n_pos)


def eval_auc(
    classifier: TFNDownstreamClassifier,
    loader: DataLoader,
    label_fn: Callable[[dict], list[int]],
    device: torch.device,
    max_batches: int | None = None,
) -> AucResult:
    preds, labels = predict(classifier, loader, label_fn, device, max_batches)
    return auc_from_preds(preds, labels)


def eval_full(
    classifier: TFNDownstreamClassifier,
    loader: DataLoader,
    label_fn: Callable[[dict], list[int]],
    device: torch.device,
    max_batches: int | None = None,
) -> MetricsResult:
    """Single forward pass that yields AUROC + AUPRC + ECE on the same preds.

    All three metrics are returned as ``None`` when the loader has a single
    class (AUROC/AUPRC undefined) or no samples (ECE undefined).
    """
    from ..evaluation.metrics import auprc, expected_calibration_error
    preds, labels = predict(classifier, loader, label_fn, device, max_batches)
    n = len(labels)
    n_pos = sum(labels)
    if not labels or len(set(labels)) < 2:
        return MetricsResult(auroc=None, auprc=None,
                              ece=expected_calibration_error(preds, labels),
                              n=n, n_pos=n_pos, preds=preds, labels=labels)
    auroc_val = float(roc_auc_score(labels, preds))
    return MetricsResult(
        preds=preds, labels=labels,
        auroc=auroc_val,
        auprc=auprc(preds, labels),
        ece=expected_calibration_error(preds, labels),
        n=n, n_pos=n_pos,
    )
