"""Houlsby-style FFN adapters for TFNFullModel.

A small bottleneck FFN placed after the LSTM encoder, before the LM head or
classifier. The base model stays frozen; only the adapter (and the head, if enabled)
trains::

    x -> Linear(hidden, bottleneck) -> GELU -> Linear(bottleneck, hidden) -> x + delta

The up-projection starts at zero, so the adapter begins as the identity (Houlsby et
al., "Parameter-Efficient Transfer Learning for NLP", ICML 2019).
"""
from __future__ import annotations


import torch
import torch.nn as nn


class FFNAdapter(nn.Module):
    """Bottleneck FFN with residual connection. Identity at initialisation."""

    def __init__(self, hidden_size: int, bottleneck: int = 64, activation: str = "gelu"):
        super().__init__()
        if bottleneck <= 0:
            raise ValueError(f"bottleneck must be > 0, got {bottleneck}")
        self.hidden_size = hidden_size
        self.bottleneck = bottleneck
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"unknown activation: {activation}")
        # Start as the identity: a zero up-projection leaves the residual unchanged.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


def attach_ffn_adapter(model, bottleneck: int = 64, *, attr: str,
                       freeze_base: bool = True) -> nn.Module:
    """Attach an FFNAdapter to ``model`` as ``model.<attr>``.

    ``attr`` is required and keyword-only, so every caller names the level
    ("cohort_adapter" or "patient_adapter"). ``TFNFullModel`` applies whatever is in
    these slots to the hidden states after the encoder. The base parameters are
    frozen.

    Calling it again on a model that already has the adapter returns the existing
    one. Safe to deepcopy.
    """
    existing = getattr(model, attr, None)
    if existing is not None:
        return existing

    device = next(model.parameters()).device
    hidden = model.n_embd  # TFNFullModel exposes this
    adapter = FFNAdapter(hidden, bottleneck=bottleneck).to(device)
    setattr(model, attr, adapter)

    if freeze_base:
        # Freeze everything that is not part of the adapter
        adapter_prefix = f"{attr}."
        for n, p in model.named_parameters():
            if not n.startswith(adapter_prefix):
                p.requires_grad_(False)
    return adapter


def adapter_state_dict(model: nn.Module, *, attr: str) -> dict[str, torch.Tensor]:
    """Only the adapter's parameters, keyed by module path.

    Lets federated and continual code send or save adapter weights without the
    frozen backbone.
    """
    adapter = getattr(model, attr, None)
    if adapter is None:
        raise ValueError(f"model has no adapter at attribute {attr!r}")
    return {f"{attr}.{n}": p.detach().clone() for n, p in adapter.named_parameters()}


def load_adapter_state_dict(model: nn.Module, state: dict[str, torch.Tensor],
                            *, attr: str) -> None:
    adapter = getattr(model, attr, None)
    if adapter is None:
        raise ValueError(f"model has no adapter at attribute {attr!r}")
    prefix = f"{attr}."
    local = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    missing = set(dict(adapter.named_parameters())) - set(local)
    if missing:
        raise KeyError(f"missing adapter weights: {sorted(missing)[:5]}")
    for n, p in adapter.named_parameters():
        p.data.copy_(local[n].to(p.device))


HEAD_PREFIX = "classifier."


def head_bias_parameters(classifier) -> list[tuple[str, "torch.nn.Parameter"]]:
    """The classifier head's bias parameters (the intercepts), as ``(name, param)``.

    The class-aware split mostly shifts base rates between centres (0.42 % to 3.10 %
    positive), and the right fix for a base-rate shift is to change the intercept,
    not the representation. With the head frozen, the adapter can only fake that by
    distorting features, which hurt ranking. Training just these few numbers makes
    the fix possible.
    """
    return [(n, p) for n, p in classifier.classifier.named_parameters()
            if n.endswith("bias")]


def level_state_dict(classifier, attr: str = "cohort_adapter",
                     include_head: bool = True,
                     head_bias_only: bool = False) -> dict[str, torch.Tensor]:
    """Everything one adaptation level learns: its adapter, plus the head if trained.

    When a level trains the classifier head, the head is part of what it learned and
    must be saved with the adapter; otherwise the adapter would later be scored
    against the population's head, which it was not trained with.

    Keys are ``<attr>.<param>`` for the adapter and ``classifier.<param>`` for the
    head, so the two stay distinguishable when deltas are aggregated across levels.
    """
    state = adapter_state_dict(classifier.backbone, attr=attr)
    if head_bias_only:
        # Only the intercepts travel: a few numbers, so the adapter-only saving is
        # kept (the whole head file is ~529 KB, bigger than the 264 KB adapter).
        for n, p in head_bias_parameters(classifier):
            state[f"{HEAD_PREFIX}{n}"] = p.detach().clone()
    elif include_head:
        for n, p in classifier.classifier.named_parameters():
            state[f"{HEAD_PREFIX}{n}"] = p.detach().clone()
    return state


def load_level_state_dict(classifier, state: dict[str, torch.Tensor],
                          attr: str = "cohort_adapter") -> None:
    """Inverse of :func:`level_state_dict`; head is optional in ``state``."""
    adapter_state = {k: v for k, v in state.items() if k.startswith(f"{attr}.")}
    if adapter_state:
        load_adapter_state_dict(classifier.backbone, adapter_state, attr=attr)
    head_state = {k[len(HEAD_PREFIX):]: v for k, v in state.items()
                  if k.startswith(HEAD_PREFIX)}
    if head_state:
        cur = dict(classifier.classifier.named_parameters())
        for n, v in head_state.items():
            if n in cur:
                cur[n].data.copy_(v.to(cur[n].device))


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
