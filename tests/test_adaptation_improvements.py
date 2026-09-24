"""Two method changes aimed at letting adaptation actually help. Both OFF by default.

**Per-hospital round selection.** Model selection scored ``macro_average`` of the
per-hospital validation AUROCs — one scalar — and restored every centre at that
one round. Under a class-aware split the centres differ materially (population
AUROC 0.9072 / 0.8300 / 0.8949), so a round chosen for the largest centre is
imposed on the others. That contradicts the project's own premise that centres
need centre-specific models, inside the step that picks the model.

**Bias-only head adaptation.** The head is frozen during adaptation, so a centre
whose positive rate is 0.42 % vs another's 3.10 % can only express that shift by
distorting features through a fixed boundary — measured as ‖Δ‖=5.05 buying
ECE 0.2527→0.1570 at a cost of 0.016 AUROC. Unfreezing only the intercepts makes
the correction expressible for a few floats per round, preserving adapter-only
transfer (the full head file is ~529 KB against the adapter's 264 KB payload).
"""
from __future__ import annotations

import torch.nn as nn

from cocofl.adapters.ffn import (
    FFNAdapter, head_bias_parameters, level_state_dict,
)
from cocofl.evaluation.baselines import _round_loop_with_early_stopping
from cocofl.hierarchical.cohort import CohortConfig
from cocofl.hierarchical.coordinator import CoordinatorConfig


# --- defaults must not change the method ---------------------------------

def test_both_changes_are_off_by_default():
    assert CohortConfig().train_head_bias_only is False
    import inspect
    sig = inspect.signature(_round_loop_with_early_stopping)
    assert sig.parameters["per_hospital_selection"].default is False


def test_bias_only_does_not_imply_training_the_whole_head():
    cfg = CohortConfig(train_head_bias_only=True)
    assert cfg.train_head is False, "bias-only must not unfreeze the weights too"


# --- bias-only: which parameters are selected -----------------------------

class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4
        self.layer = nn.Linear(4, 4)


class _FakeClf(nn.Module):
    """Minimal stand-in: a head plus a backbone carrying a real FFN adapter,
    since ``level_state_dict`` reads both namespaces."""

    def __init__(self, with_adapter: bool = True):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
        self.backbone = _FakeBackbone()
        if with_adapter:
            self.backbone.cohort_adapter = FFNAdapter(hidden_size=4, bottleneck=2)


def test_head_bias_parameters_selects_only_biases():
    clf = _FakeClf()
    names = [n for n, _ in head_bias_parameters(clf)]
    assert names == ["0.bias", "2.bias"]
    assert all(n.endswith("bias") for n in names)


def test_bias_only_is_far_cheaper_than_the_full_head():
    clf = _FakeClf()
    bias = sum(p.numel() for _, p in head_bias_parameters(clf))
    full = sum(p.numel() for p in clf.classifier.parameters())
    assert bias < full, "intercepts only"
    assert bias == 4, "3 + 1 bias terms"
    assert full == 4 + 12 + 3, "plus both weight matrices"


def test_state_dict_carries_only_biases_under_bias_only():
    clf = _FakeClf()
    st = level_state_dict(clf, attr="cohort_adapter",
                          include_head=False, head_bias_only=True)
    head_keys = [k for k in st if k.startswith("classifier.")]
    assert head_keys, "the intercepts must travel, or the fix does nothing"
    assert all(k.endswith("bias") for k in head_keys), head_keys


def test_state_dict_head_is_absent_when_everything_is_frozen():
    clf = _FakeClf()
    st = level_state_dict(clf, attr="cohort_adapter",
                          include_head=False, head_bias_only=False)
    assert not [k for k in st if k.startswith("classifier.")]


def test_full_head_still_available_when_explicitly_requested():
    clf = _FakeClf()
    st = level_state_dict(clf, attr="cohort_adapter", include_head=True)
    head_keys = {k for k in st if k.startswith("classifier.")}
    assert any(k.endswith("weight") for k in head_keys), "unchanged legacy path"


# --- per-hospital selection: the semantics that make it safe --------------

def _pick(per_hosp_val_by_round, use_per_hospital):
    """Mirrors the harness: round 0 is always a candidate."""
    hids = per_hosp_val_by_round[0].keys()
    if use_per_hospital:
        return {h: max(range(len(per_hosp_val_by_round)),
                       key=lambda r: per_hosp_val_by_round[r][h]) for h in hids}
    macro = [sum(rv.values()) / len(rv) for rv in per_hosp_val_by_round]
    best = max(range(len(macro)), key=lambda r: macro[r])
    return {h: best for h in hids}


# Round 1 wins on macro (0.725 > 0.700) because hospital 0 gains more than
# hospital 1 loses — the exact situation where one shared round is wrong.
_SPLIT_PEAKS = [{0: 0.80, 1: 0.60}, {0: 0.95, 1: 0.50}]


def test_global_selection_imposes_one_round_on_every_centre():
    assert _pick(_SPLIT_PEAKS, use_per_hospital=False) == {0: 1, 1: 1}, (
        "hospital 1 is dragged to round 1 even though round 0 was better for it"
    )


def test_per_hospital_selection_lets_centres_disagree():
    assert _pick(_SPLIT_PEAKS, use_per_hospital=True) == {0: 1, 1: 0}


def test_per_hospital_selection_beats_global_for_the_dragged_centre():
    glob = _pick(_SPLIT_PEAKS, use_per_hospital=False)
    per = _pick(_SPLIT_PEAKS, use_per_hospital=True)
    for h in (0, 1):
        assert _SPLIT_PEAKS[per[h]][h] >= _SPLIT_PEAKS[glob[h]][h]
    assert _SPLIT_PEAKS[per[1]][1] > _SPLIT_PEAKS[glob[1]][1], "h1 strictly better"


def test_per_hospital_selection_never_lands_below_a_centres_round_0():
    """The safety property: round 0 is a candidate for every centre separately."""
    rounds = [{0: 0.80, 1: 0.70}, {0: 0.10, 1: 0.05}, {0: 0.20, 1: 0.02}]
    picked = _pick(rounds, use_per_hospital=True)
    for h, r in picked.items():
        assert rounds[r][h] >= rounds[0][h]


def test_the_two_agree_when_centres_peak_together():
    rounds = [{0: 0.50, 1: 0.50}, {0: 0.90, 1: 0.90}]
    assert _pick(rounds, True) == _pick(rounds, False)


def test_coordinator_config_still_defaults_to_shared_weighting():
    """Guard against these edits changing unrelated defaults."""
    c = CoordinatorConfig()
    assert c.patient_weighting_scheme is None and c.cohort_weighting_scheme is None
