"""Per-patient personalisation strategies for TFNFullModel with FFN adapters.

Used by the standalone personalisation ablation (``cocofl_personalize``):

* ``personalize_full``: full fine-tune (A0, the forgetting reference). *
``personalize_head_only``: backbone frozen, train only the MLP head (A1). *
``personalize_ffn``: backbone frozen, train an FFN adapter plus the head (A2).

The continual-learning variants (A3 EWC, A4 replay, A5 distillation) are in
``cocofl.personalisation.continual`` and reuse ``personalize_ffn`` through its
``extra_loss_fn`` hook.

Each batch is ``((x, elapsed, cat, num), y_dict)``, as produced by ``make_collate``:

- x, elapsed: (B, T) long, float - cat: (B, n_cat) long - num: (B, n_num) float -
y_dict: {"expected": list[str], "patient_id": list[int]}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.classifier import TFNDownstreamClassifier
from ..adapters import attach_ffn_adapter


# Which parameters train

def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(True)


def set_trainable_head_only(classifier: TFNDownstreamClassifier) -> None:
    freeze_all(classifier)
    for p in classifier.classifier.parameters():
        p.requires_grad_(True)


def set_trainable_ffn_plus_head(classifier: TFNDownstreamClassifier,
                                attr: str = "ffn_adapter") -> None:
    """Only the adapter + classifier head are trainable; backbone stays frozen."""
    freeze_all(classifier)
    adapter = getattr(classifier.backbone, attr, None)
    if adapter is None:
        raise ValueError(f"classifier.backbone has no adapter at {attr!r} — call "
                         "attach_ffn_adapter first")
    for p in adapter.parameters():
        p.requires_grad_(True)
    for p in classifier.classifier.parameters():
        p.requires_grad_(True)


@dataclass
class PersonalizeConfig:
    n_iters: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 50


# Losses

def _bce_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 1 or logits.size(-1) == 1:
        return F.binary_cross_entropy_with_logits(logits.view(-1), labels.float())
    raise ValueError(f"expected 1-d logits for binary task, got shape {tuple(logits.shape)}")


def _infinite(loader: Iterable) -> Iterator:
    while True:
        for batch in loader:
            if batch is None:
                continue
            yield batch


# Input-only wrapper for the continual-learning strategies

def make_input_only_classifier(classifier: TFNDownstreamClassifier,
                                cat: torch.Tensor, num: torch.Tensor):
    """Return ``fn(x_long) -> logits``, hiding TFN's extra arguments.

    Lets the continual-learning strategies call ``classifier(x)``. The static
    features are fixed from the batch, and the time gaps are set to a constant 1, so
    the time decay is not exercised. This is only used by the standalone ablation,
    not by the reported grid.
    """
    def fn(x: torch.Tensor) -> torch.Tensor:
        elapsed = torch.ones_like(x, dtype=torch.float32)
        return classifier(
            input_ids=x, elapsed_times=elapsed,
            static_categorical=cat, static_numerical=num,
        )
    return fn


# Training loop

ExtraLossFn = Callable[
    [Callable, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


def _train_classifier(
    classifier: TFNDownstreamClassifier,
    loader: DataLoader,
    label_fn: Callable[[dict], list[int]],
    cfg: PersonalizeConfig,
    device: torch.device,
    extra_loss_fn: ExtraLossFn | None = None,
) -> list[float]:
    """Generic training loop.

    ``extra_loss_fn(clf_fn, x, y, base_loss)`` lets a CL method add a regulariser.
    Its first argument is an input-only callable (see
    ``make_input_only_classifier``).
    """
    trainable = [p for p in classifier.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters — set_trainable_*() first")

    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    classifier.train()
    history: list[float] = []
    it = _infinite(loader)
    for step in range(cfg.n_iters):
        batch = next(it)
        x_tuple, y_dict = batch
        x, elapsed, cat, num = (t.to(device, non_blocking=True) for t in x_tuple[:4])
        y = torch.tensor(label_fn(y_dict), dtype=torch.float32, device=device)

        logits = classifier(
            input_ids=x, elapsed_times=elapsed,
            static_categorical=cat, static_numerical=num,
        )
        loss = _bce_logits(logits, y)
        if extra_loss_fn is not None:
            clf_fn = make_input_only_classifier(classifier, cat, num)
            loss = extra_loss_fn(clf_fn, x, y, loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        optimizer.step()

        history.append(float(loss.detach()))
        if (step + 1) % cfg.log_every == 0:
            recent = sum(history[-cfg.log_every:]) / cfg.log_every
            print(f"  step {step + 1:>4}/{cfg.n_iters}  loss={recent:.4f}")
    return history


# Strategies

def personalize_head_only(
    classifier: TFNDownstreamClassifier,
    loader: DataLoader,
    label_fn: Callable[[dict], list[int]],
    cfg: PersonalizeConfig,
    device: torch.device,
    extra_loss_fn: ExtraLossFn | None = None,
) -> list[float]:
    """A1: backbone frozen, head only."""
    set_trainable_head_only(classifier)
    return _train_classifier(classifier, loader, label_fn, cfg, device, extra_loss_fn)


def personalize_ffn(
    classifier: TFNDownstreamClassifier,
    loader: DataLoader,
    label_fn: Callable[[dict], list[int]],
    cfg: PersonalizeConfig,
    device: torch.device,
    bottleneck: int = 64,
    extra_loss_fn: ExtraLossFn | None = None,
) -> list[float]:
    """A2: an FFN adapter on the frozen backbone, with the classifier head trainable.

    Attaches an FFNAdapter to ``classifier.backbone`` unless one is already there.
    The base model's weights stay frozen.
    """
    attach_ffn_adapter(classifier.backbone, bottleneck=bottleneck,
                       attr="patient_adapter")
    set_trainable_ffn_plus_head(classifier)
    return _train_classifier(classifier, loader, label_fn, cfg, device, extra_loss_fn)


def personalize_full(
    classifier: TFNDownstreamClassifier,
    loader: DataLoader,
    label_fn: Callable[[dict], list[int]],
    cfg: PersonalizeConfig,
    device: torch.device,
    extra_loss_fn: ExtraLossFn | None = None,
) -> list[float]:
    """A0: full fine-tune (the forgetting reference)."""
    unfreeze_all(classifier)
    return _train_classifier(classifier, loader, label_fn, cfg, device, extra_loss_fn)


# Grouping windows by patient

# Keyed by id(dataset). The dataset is stored in the value too, so its id cannot be
# reused by another object while the entry exists.
_PATIENT_BUCKET_CACHE: dict[int, tuple[object, dict[int, list[int]]]] = {}


def per_patient_subsets(dataset, min_samples: int = 1) -> dict[int, list[int]]:
    """Group dataset indices by ``patient_id``.

    Works with datasets returning ``(x, y_dict)`` with ``y_dict["patient_id"]``.
    Grouping walks the whole dataset (~490 K windows on MIMIC) and the harness calls
    it once per hospital per alpha, so the result is cached per dataset;
    ``min_samples`` is applied afterwards because it only filters.

    When the dataset has ``start_indices`` and ``patient_id_at_idx``, patient ids
    are read from those arrays instead of building a 2048-token window per index.
    Unlike the ``dataset[idx]`` path, the array path cannot skip indices whose
    ``__getitem__`` raises; in the harness such an index would fail the run anyway.
    """
    cached = _PATIENT_BUCKET_CACHE.get(id(dataset))
    if cached is not None:
        buckets = cached[1]
    else:
        buckets = {}
        start_indices = getattr(dataset, "start_indices", None)
        pid_at_idx = getattr(dataset, "patient_id_at_idx", None)
        pids = None
        if start_indices is not None and pid_at_idx is not None:
            # patient_id_at_idx is a lazy lookup on the sharded backend, so pass all
            # indices in one call instead of looping 490 K times.
            try:
                pids = pid_at_idx[start_indices].reshape(-1).tolist()
            except (TypeError, IndexError, NotImplementedError):
                pids = None
        if pids is not None:
            for idx, pid in enumerate(pids):
                buckets.setdefault(int(pid), []).append(idx)
        else:
            for idx in range(len(dataset)):
                try:
                    _, y = dataset[idx]
                except Exception:
                    continue
                pid = y.get("patient_id")
                if pid is None:
                    continue
                buckets.setdefault(int(pid), []).append(idx)
        _PATIENT_BUCKET_CACHE[id(dataset)] = (dataset, buckets)
    return {pid: ix for pid, ix in buckets.items() if len(ix) >= min_samples}


def split_indices_by_time(dataset, indices: list[int], train_frac: float = 0.5
                          ) -> tuple[list[int], list[int]]:
    """Sort ``indices`` by time and split them into early and late at ``train_frac``."""
    # Read prediction times from the dataset's arrays when possible; ``dataset[i]``
    # builds a whole window just to read one timestamp.
    starts = getattr(dataset, "start_indices", None)
    ts = getattr(dataset, "times", None)
    if starts is not None and ts is not None:
        times = [(int(ts[starts[i]]), i) for i in indices]
    else:
        times = []
        for i in indices:
            _, y = dataset[i]
            times.append((y.get("prediction_time", i), i))
    times.sort()
    n_train = max(1, int(len(times) * train_frac))
    return [i for _, i in times[:n_train]], [i for _, i in times[n_train:]]
