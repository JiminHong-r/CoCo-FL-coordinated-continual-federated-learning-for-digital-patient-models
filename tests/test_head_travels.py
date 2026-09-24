"""Whatever a level trains must be what that level persists and is scored with.

Cohort and patient training explicitly unfreeze ``classifier.classifier``
("Make sure classifier head is also trainable for downstream"), so the head is
part of what those levels learn. But the state that survived a round was
adapter-only, and evaluation rebuilt the model from the *population* classifier
— so the adapter was scored against a head it had never been co-adapted with,
every round. FedAvg was unaffected: ``federated_state_dict(include_head=True)``
already carried its head, which made the headline "FedAvg beats the hierarchy"
comparison unfair rather than merely unfavourable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from cocofl.adapters import (
    attach_ffn_adapter, level_state_dict, load_level_state_dict,
)
from cocofl.hierarchical.upward_prop import propagate_patients_to_cohort


class _Backbone(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.n_embd = hidden
        self.lin = nn.Linear(hidden, hidden)


class _Clf(nn.Module):
    """Minimal stand-in for TFNDownstreamClassifier's (backbone, classifier) shape."""

    def __init__(self, hidden=8):
        super().__init__()
        self.backbone = _Backbone(hidden)
        self.classifier = nn.Sequential(nn.Linear(hidden, 4), nn.ReLU(), nn.Linear(4, 1))


def _make(attr="cohort_adapter"):
    clf = _Clf()
    attach_ffn_adapter(clf.backbone, bottleneck=2, attr=attr)
    return clf


def test_level_state_includes_both_adapter_and_head():
    state = level_state_dict(_make(), attr="cohort_adapter")
    assert any(k.startswith("cohort_adapter.") for k in state), "adapter missing"
    assert any(k.startswith("classifier.") for k in state), "head missing"


def test_head_round_trips_through_save_and_load():
    src = _make()
    with torch.no_grad():
        for p in src.classifier.parameters():
            p.fill_(0.5)
    state = level_state_dict(src, attr="cohort_adapter")

    dst = _make()
    with torch.no_grad():
        for p in dst.classifier.parameters():
            p.zero_()
    load_level_state_dict(dst, state, attr="cohort_adapter")

    for p in dst.classifier.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.5)), (
            "a trained head must survive the round it was trained in"
        )


def test_adapter_only_state_still_loads():
    """Backwards compatible: a head-less state must not raise."""
    src = _make()
    state = {k: v for k, v in level_state_dict(src, attr="cohort_adapter").items()
             if not k.startswith("classifier.")}
    load_level_state_dict(_make(), state, attr="cohort_adapter")


def test_head_deltas_propagate_upward():
    """Head keys are namespace-free; the adapter-renaming branch would drop them."""
    cohort = {
        "cohort_adapter.up.weight": torch.zeros(4, 2),
        "classifier.0.weight": torch.zeros(4, 8),
    }
    delta = {
        "patient_adapter.up.weight": torch.ones(4, 2),
        "classifier.0.weight": torch.ones(4, 8),
    }
    out = propagate_patients_to_cohort(cohort, [delta], [1], eta_cohort=0.5)

    assert torch.allclose(out["cohort_adapter.up.weight"], torch.full((4, 2), 0.5))
    assert torch.allclose(out["classifier.0.weight"], torch.full((4, 8), 0.5)), (
        "head deltas must reach the cohort, not be silently skipped"
    )


def test_head_shape_mismatch_is_skipped_not_crashed():
    cohort = {"classifier.0.weight": torch.zeros(4, 8)}
    delta = {"classifier.0.weight": torch.ones(3, 8)}
    out = propagate_patients_to_cohort(cohort, [delta], [1], eta_cohort=0.5)
    assert torch.allclose(out["classifier.0.weight"], torch.zeros(4, 8))
