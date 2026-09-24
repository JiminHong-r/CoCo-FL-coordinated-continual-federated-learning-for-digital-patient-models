"""Houlsby-style FFN adapters for TFNFullModel.

A small bottleneck FFN inserted at the output of the LSTM encoder (before the
LM head / classifier). The base model is frozen during personalisation; only
the adapter (and the classifier head, if used) trains.

Adapter:
    x -> Linear(hidden, bottleneck) -> GELU -> Linear(bottleneck, hidden) -> x + delta

The up-projection is initialised to zero so the adapter starts as an identity
function. This matches the Houlsby et al. ("Parameter-Efficient Transfer
Learning for NLP", ICML 2019) recipe.
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
        # Identity-at-init: up-projection is zero so the residual passes through unchanged.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


def attach_ffn_adapter(model, bottleneck: int = 64, *, attr: str,
                       freeze_base: bool = True) -> nn.Module:
    """Attach an FFNAdapter to ``model`` as ``model.<attr>``.

    ``attr`` is REQUIRED and keyword-only: it used to default to the legacy
    ``ffn_adapter`` slot, so a caller that forgot to name a level silently got a
    third adapter applied after cohort and patient. There are only two levels.

    Sets the adapter into the model's pre-declared slot. ``TFNFullModel`` checks
    this slot in its forward pass and applies the adapter to the post-encoder
    hidden states. The base parameters are frozen by default.

    Idempotent: re-calling on an already-wrapped model returns the existing
    adapter (no overwrite). Safe under deepcopy because no closures capture
    references to the original model.
    """
    existing = getattr(model, attr, None)
    if existing is not None:
        return existing

    device = next(model.parameters()).device
    hidden = model.n_embd  # TFNFullModel exposes this
    adapter = FFNAdapter(hidden, bottleneck=bottleneck).to(device)
    setattr(model, attr, adapter)

    if freeze_base:
        # Freeze everything that isn't part of the adapter
        adapter_prefix = f"{attr}."
        for n, p in model.named_parameters():
            if not n.startswith(adapter_prefix):
                p.requires_grad_(False)
    return adapter


def adapter_state_dict(model: nn.Module, *, attr: str) -> dict[str, torch.Tensor]:
    """Return only the adapter A/B parameters keyed by their module path.

    Used by federated / continual code to transmit/save adapter weights without
    the entire (frozen) backbone.
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
    """The classifier head's bias parameters, as ``(name, param)``.

    These are the intercepts. Under a label shift — which is what a class-aware
    federated split creates, with per-centre positive rates spanning 0.42 % to
    3.10 % — the Bayes-optimal correction is a change to the intercept, not to
    the representation. With the head fully frozen the only way to express that
    correction is to distort features through a fixed decision boundary, which
    is measurably worse for ranking (local_finetune: ‖Δ‖=5.05, ECE 0.253→0.157,
    AUROC −0.016). Unfreezing just these makes the correction expressible at a
    cost of a handful of floats per round.
    """
    return [(n, p) for n, p in classifier.classifier.named_parameters()
            if n.endswith("bias")]


def level_state_dict(classifier, attr: str = "cohort_adapter",
                     include_head: bool = True,
                     head_bias_only: bool = False) -> dict[str, torch.Tensor]:
    """Everything one adaptation level learns: its adapter, plus the head.

    Cohort and patient training unfreeze ``classifier.classifier`` (see
    ``coordinator._train_cohorts`` and ``patient.build_patient_classifier``), so
    the head is part of what a level actually learns. Persisting only the
    adapter meant the trained head was discarded at the end of every round and
    the adapter was then evaluated against the *population's* head — a head it
    was never co-adapted with. FedAvg never had this problem because
    ``federated_state_dict`` already carried its head.

    Keys are ``<attr>.<param>`` for the adapter and ``classifier.<param>`` for
    the head, so the two namespaces stay distinguishable when deltas are
    aggregated across levels.
    """
    state = adapter_state_dict(classifier.backbone, attr=attr)
    if head_bias_only:
        # Only the intercepts travel. Communication cost is a few floats, so
        # adapter-only transfer (and the 162.5x saving) is preserved — the whole
        # saved head file is ~529 KB, larger than the 264 KB adapter payload.
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
