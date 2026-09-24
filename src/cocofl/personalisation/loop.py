"""Per-patient personalisation strategies for TFNFullModel + FFNAdapter.

Three core strategies (the per-patient personalisation loop which targets the legacy
TM-LSTM + LoRA pipeline):

* ``personalize_full``        — full fine-tune (A0, reference for catastrophic forgetting).
* ``personalize_head_only``   — freeze backbone; train MLP head only (A1).
* ``personalize_ffn``         — freeze backbone; train Houlsby FFN adapter + head (A2).

The CL-augmented variants (A3 EWC, A4 Replay, A5 Distill) live in
``cocofl.personalisation.continual`` and reuse ``personalize_ffn`` via the
``extra_loss_fn`` hook.

Loader contract: each batch is ``((x, elapsed, cat, num), y_dict)`` where:
  - x, elapsed: (B, T) long, float
  - cat: (B, n_cat) long
  - num: (B, n_num) float
  - y_dict: {"expected": list[str], "patient_id": list[int]}

This matches the collate produced by ``downstream.finetune.make_collate``.
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


# --- Trainable-set helpers ---------------------------------------------------

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


# --- Loss helpers ------------------------------------------------------------

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


# --- Closure: input-only callable for continual-learning strategies ----------

def make_input_only_classifier(classifier: TFNDownstreamClassifier,
                                cat: torch.Tensor, num: torch.Tensor):
    """Return a callable ``fn(x_long) -> logits`` that hides TFN's extra args.

    Used so that the continual-learning strategies (EWC / Replay / Distill),
    which historically expect ``classifier(x)``, work unchanged. Static features
    are baked in from the batch; elapsed_times defaults to ones (downstream
    datasets don't expose per-token timing).
    """
    def fn(x: torch.Tensor) -> torch.Tensor:
        elapsed = torch.ones_like(x, dtype=torch.float32)
        return classifier(
            input_ids=x, elapsed_times=elapsed,
            static_categorical=cat, static_numerical=num,
        )
    return fn


# --- Core training loop ------------------------------------------------------

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
    """Generic training loop. ``extra_loss_fn(clf_fn, x, y, base_loss)`` lets CL methods
    add a regulariser. The first argument is an *input-only* callable (see
    ``make_input_only_classifier``) so existing CL code stays unchanged."""
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


# --- Public strategies -------------------------------------------------------

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
    """A2: Houlsby FFN adapter on top of the frozen backbone + classifier head trainable.

    Attaches an FFNAdapter to ``classifier.backbone`` if not already present
    (idempotent). The base model's weights are frozen.
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
    """A0: full fine-tune (catastrophic forgetting reference)."""
    unfreeze_all(classifier)
    return _train_classifier(classifier, loader, label_fn, cfg, device, extra_loss_fn)


# --- Patient cohort utilities ------------------------------------------------

# Keyed by ``id(dataset)``; the dataset itself is kept in the value so the id
# cannot be recycled onto a different object while the entry is live.
_PATIENT_BUCKET_CACHE: dict[int, tuple[object, dict[int, list[int]]]] = {}


def per_patient_subsets(dataset, min_samples: int = 1) -> dict[int, list[int]]:
    """Group dataset indices by ``patient_id``. Datasets returning
    ``(x, y_dict)`` with ``y_dict["patient_id"]`` set are compatible.

    This walks the whole dataset (~490 K windows for the MIMIC tasks) and the
    WP5 harness calls it once per hospital per α, so the unfiltered grouping is
    memoised per dataset instance — ``min_samples`` is applied to the cached
    result, since it only filters.

    On datasets exposing ``start_indices``/``patient_id_at_idx`` the patient id
    is read straight off those arrays instead of building a 2048-token window
    per index just to read one label field.

    One behavioural difference worth knowing: the ``dataset[idx]`` path silently
    drops any index whose ``__getitem__`` raises, and the array path cannot see
    those. This is a no-op for the WP5 harness — the cohort train loader and the
    eval loaders iterate the same indices through ``__getitem__`` anyway, so an
    index that raises would fail the run outright rather than be quietly
    excluded from patient streams. It matters only for a dataset that is
    partially unreadable by design.
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
            # ``patient_id_at_idx`` is a lazy per-index lookup on the sharded
            # backend; hand it the whole index tensor in one call rather than
            # paying Python-loop overhead 490 K times.
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
    """Order ``indices`` chronologically and split early/late at ``train_frac``."""
    # Read prediction times off the dataset's own arrays when possible; the
    # ``dataset[i]`` fallback builds a full 2048-token window per index just to
    # read one timestamp.
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
