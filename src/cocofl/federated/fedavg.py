"""FedAvg simulation for hospital-federated FFN adapter training (WP4).

Each round:
  1. Server broadcasts the current adapter (+ head) state to all hospitals.
  2. Each hospital trains locally for ``local_epochs`` epochs on its own slice.
  3. Hospitals upload their updated adapter (+ head) state to the server.
  4. Server averages weighted by hospital sample count.

The TFN backbone is frozen throughout — only adapter (+ optional classifier
head) parameters travel across the network.

This is a *simulation* — all hospitals run in the same process sequentially
on a single GPU. For an actual deployment, replace ``run_round`` with the
Flower client/server runtime.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..models.classifier import TFNDownstreamClassifier
from ..adapters import (
    adapter_state_dict, attach_ffn_adapter, load_adapter_state_dict,
)


# ---- Federated state plumbing ----------------------------------------------

def federated_state_dict(classifier: TFNDownstreamClassifier,
                          include_head: bool = True,
                          adapter_attr: str = "ffn_adapter") -> dict[str, torch.Tensor]:
    """Snapshot the parameters that travel between hospitals and server.

    Always includes the adapter; optionally includes the classifier MLP head.
    """
    state = adapter_state_dict(classifier.backbone, attr=adapter_attr)
    if include_head:
        for n, p in classifier.classifier.named_parameters():
            state[f"classifier.{n}"] = p.detach().clone()
    return state


def load_federated_state(classifier: TFNDownstreamClassifier,
                          state: dict[str, torch.Tensor],
                          adapter_attr: str = "ffn_adapter") -> None:
    """Load adapter + head state in-place."""
    adapter_prefix = f"{adapter_attr}."
    adapter_state = {k: v for k, v in state.items() if k.startswith(adapter_prefix)}
    head_state = {k[len("classifier."):]: v for k, v in state.items()
                  if k.startswith("classifier.")}
    if adapter_state:
        load_adapter_state_dict(classifier.backbone, adapter_state, attr=adapter_attr)
    if head_state:
        cur = dict(classifier.classifier.named_parameters())
        for n, v in head_state.items():
            cur[n].data.copy_(v.to(cur[n].device))


def fedavg_states(states: list[dict[str, torch.Tensor]],
                   weights: list[float]) -> dict[str, torch.Tensor]:
    """Weighted average of state dicts. ``weights`` are absolute sizes."""
    if not states:
        raise ValueError("fedavg_states: no states to average")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("fedavg_states: total weight must be positive")
    norm = [w / total for w in weights]
    avg: dict[str, torch.Tensor] = {}
    keys = set(states[0].keys())
    for st in states[1:]:
        if set(st.keys()) != keys:
            raise ValueError("fedavg_states: state dicts must share keys")
    for k in keys:
        stacked = torch.stack([s[k] for s in states])
        w = torch.tensor(norm, dtype=stacked.dtype, device=stacked.device)
        avg[k] = (stacked * w.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)
    return avg


# ---- Local training step ----------------------------------------------------

@dataclass
class FedRoundConfig:
    local_epochs: int = 1
    local_steps_per_epoch: int | None = None  # None => one full pass over loader
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    include_head_in_fedavg: bool = True


def _local_train_one_hospital(
    classifier: TFNDownstreamClassifier,
    loader: DataLoader,
    label_fn: Callable[[dict], list[int]],
    cfg: FedRoundConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    """Downstream local training: adapter (+ head) trained with BCE on task labels."""
    classifier.train()
    trainable = [p for p in classifier.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable params — call attach_ffn_adapter + freeze first")
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_seen = 0
    for epoch in range(cfg.local_epochs):
        steps = 0
        for batch in loader:
            if batch is None:
                continue
            x_tuple, y_dict = batch
            x, elapsed, cat, num = (t.to(device) for t in x_tuple[:4])
            y = torch.tensor(label_fn(y_dict), dtype=torch.float32, device=device)
            logits = classifier(
                input_ids=x, elapsed_times=elapsed,
                static_categorical=cat, static_numerical=num,
            ).view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            n_seen += x.size(0)
            steps += 1
            if cfg.local_steps_per_epoch and steps >= cfg.local_steps_per_epoch:
                break
    state = federated_state_dict(classifier, include_head=cfg.include_head_in_fedavg)
    return state, n_seen


def _local_train_one_hospital_lm(
    classifier: TFNDownstreamClassifier,
    loader: DataLoader,
    cfg: FedRoundConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    """LM local training: adapter trained with next-token CE on token sequences.

    Loader yields TFN-format batches ``((x, elapsed, cat, num), labels_tensor)``
    where ``labels_tensor`` is the shifted next-token target (TimelineDataset.__getitem__
    output style). The classifier's underlying backbone (``classifier.backbone``)
    computes the LM loss directly via its ``labels=...`` kwarg.

    Only the adapter is trained — the classifier head is irrelevant to LM loss
    and is excluded from FedAvg by overriding ``include_head_in_fedavg=False``
    inside this function (we still return the head state for compatibility but
    it never receives a gradient).
    """
    classifier.train()
    trainable = [p for p in classifier.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable params — call attach_ffn_adapter + freeze first")
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_seen = 0
    for epoch in range(cfg.local_epochs):
        steps = 0
        for batch in loader:
            if batch is None:
                continue
            x_tuple, y = batch
            x, elapsed, cat, num = (t.to(device) for t in x_tuple[:4])
            y = y.to(device)
            # Use the backbone directly — LM CE loss via labels kwarg
            out = classifier.backbone(
                input_ids=x, elapsed_times=elapsed,
                static_categorical=cat, static_numerical=num, labels=y,
            )
            loss = out.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            n_seen += x.size(0)
            steps += 1
            if cfg.local_steps_per_epoch and steps >= cfg.local_steps_per_epoch:
                break
    # For LM phase only the adapter changes — exclude the classifier head from the snapshot.
    state = federated_state_dict(classifier, include_head=False)
    return state, n_seen


# ---- Round orchestration ----------------------------------------------------

@dataclass
class FederatedRun:
    rounds: int
    hospital_loaders: dict[int, DataLoader]
    round_cfg: FedRoundConfig
    # Optional per-round callbacks
    on_round_end: Callable[[int, dict], None] | None = None
    history: list[dict] = field(default_factory=list)


def run_round(
    server_state: dict[str, torch.Tensor],
    server_template: TFNDownstreamClassifier,
    hospital_loaders: dict[int, DataLoader],
    round_cfg: FedRoundConfig,
    device: torch.device,
    objective: str = "downstream",
    label_fn: Callable[[dict], list[int]] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[int, int]]:
    """Run one FedAvg round.

    Parameters
    ----------
    objective : "downstream" or "lm"
        - "downstream" : BCE on task labels (requires ``label_fn``)
        - "lm"         : next-token CE on the timeline (loader yields LM batches)
    """
    if objective == "downstream" and label_fn is None:
        raise ValueError("objective='downstream' requires label_fn")

    states: list[dict[str, torch.Tensor]] = []
    weights: list[float] = []
    counts: dict[int, int] = {}
    for hid, loader in hospital_loaders.items():
        local_clf = copy.deepcopy(server_template)
        load_federated_state(local_clf, server_state)
        if objective == "downstream":
            state, n_seen = _local_train_one_hospital(local_clf, loader, label_fn,
                                                       round_cfg, device)
        elif objective == "lm":
            state, n_seen = _local_train_one_hospital_lm(local_clf, loader,
                                                          round_cfg, device)
        else:
            raise ValueError(f"unknown objective: {objective!r}")
        states.append(state)
        weights.append(float(max(n_seen, 1)))
        counts[hid] = n_seen
        del local_clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    new_state = fedavg_states(states, weights)
    return new_state, counts


def initialize_server_state(template: TFNDownstreamClassifier,
                             bottleneck: int = 64,
                             include_head: bool = True) -> dict[str, torch.Tensor]:
    """Attach FFN adapter on ``template`` (in place) and snapshot its initial state."""
    # Flat FedAvg keeps ONE shared adapter across centres, i.e. a single
    # cohort-level adapter — there is no per-patient level in this baseline.
    attach_ffn_adapter(template.backbone, bottleneck=bottleneck, attr="cohort_adapter")
    return federated_state_dict(template, include_head=include_head)
