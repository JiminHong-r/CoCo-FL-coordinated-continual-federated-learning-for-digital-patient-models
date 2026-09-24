"""The WP4 task 3 and 5 settings must reach the coordinator and change something.

Two past failures looked fine in the results:

* ``weighting_scheme`` and ``divergence_threshold`` were never read by the harness,
so every run used volume weighting and the frequency trigger. * ``uncertainty``
weighting got a fixed loss of 1.0 for every patient, so its inverse-loss weights
were all equal: it was ``uniform`` under another name.
"""
from __future__ import annotations

import pytest
import torch

from cocofl.hierarchical.upward_prop import _weights_for


def test_volume_weights_track_counts():
    assert _weights_for("volume", [1, 3, 6]) == [1.0, 3.0, 6.0]


def test_uniform_weights_ignore_counts():
    assert _weights_for("uniform", [1, 3, 6]) == [1.0, 1.0, 1.0]


def test_recency_without_timestamps_is_the_deprecated_position_fallback():
    """Without timestamps, recency falls back to decay by list position, and must warn.

    Position is not recency (every update in a round is equally recent), so the
    fallback must not be silent. Time-based recency is tested in
    tests/test_wp4_spec_compliance.py.
    """
    with pytest.warns(UserWarning, match="POSITION decay"):
        w = _weights_for("recency", [1, 1, 1], recency_decay=0.5)
    assert w[-1] == 1.0, "last in list gets full weight"
    assert w[0] < w[1] < w[2], "earlier positions discounted"


def test_uncertainty_is_inverse_loss():
    w = _weights_for("uncertainty", [1, 1], losses=[0.1, 1.0])
    assert w[0] > w[1], "lower loss -> higher weight"
    assert w[0] == pytest.approx(10.0)


def test_uncertainty_reaches_the_cohort_to_population_step():
    """The cohort-to-population step must receive losses too.

    Without them every ``weighting_scheme=uncertainty`` run failed at the first
    population sync with "uncertainty weighting requires losses".
    """
    from cocofl.hierarchical.upward_prop import propagate_cohorts_to_population

    cohorts = [{"a": torch.tensor([1.0, 0.0])}, {"a": torch.tensor([0.0, 1.0])}]
    out = propagate_cohorts_to_population(
        None, cohorts, [10, 10], eta_population=1.0,
        scheme="uncertainty", losses=[0.1, 1.0],
    )
    # Cohort 0 has the lower loss, so it counts more.
    assert out["a"][0] > out["a"][1]

    with pytest.raises(ValueError, match="requires losses"):
        propagate_cohorts_to_population(
            None, cohorts, [10, 10], scheme="uncertainty",
        )


def test_coordinator_passes_losses_to_the_population_step():
    """Check the call site passes losses, not only the function that receives them."""
    import inspect
    import re

    from cocofl.hierarchical.coordinator import HierarchicalCoordinator

    # Step 4 is in _finish_round, which run_round calls at the end.
    src = inspect.getsource(HierarchicalCoordinator._finish_round)
    call = re.search(
        r"propagate_cohorts_to_population\((.*?)\)\n", src, re.DOTALL,
    )
    assert call, "expected a cohort->population propagation call"
    assert "losses=" in call.group(1), (
        "cohort->population propagation must pass losses, or "
        "weighting_scheme=uncertainty raises"
    )


def test_uncertainty_with_constant_losses_degenerates_to_uniform():
    """Constant losses make uncertainty weighting identical to uniform weighting.

    If a constant loss comes back, the uncertainty arm silently duplicates the
    uniform arm.
    """
    constant = _weights_for("uncertainty", [1, 5, 9], losses=[1.0, 1.0, 1.0])
    uniform = _weights_for("uniform", [1, 5, 9])
    assert constant == uniform, "constant losses cannot distinguish the schemes"

    real = _weights_for("uncertainty", [1, 5, 9], losses=[0.2, 0.9, 0.4])
    assert real != uniform, "real losses must produce non-uniform weights"


def test_uncertainty_requires_losses():
    with pytest.raises(ValueError, match="requires losses"):
        _weights_for("uncertainty", [1, 2], losses=None)


def test_unknown_scheme_is_rejected():
    with pytest.raises(ValueError, match="unknown weighting scheme"):
        _weights_for("magic", [1, 2])


def test_patient_trainer_returns_its_loss():
    """The patient trainer must return a real per-patient loss for uncertainty weighting."""
    import inspect

    from cocofl.hierarchical.patient import train_patient_adapter

    src = inspect.getsource(train_patient_adapter)
    assert "return clf, final, delta, avg_loss" in src, (
        "train_patient_adapter must report its mean task loss"
    )


def test_coordinator_config_accepts_both_wp4_knobs():
    from cocofl.hierarchical.coordinator import CoordinatorConfig

    cfg = CoordinatorConfig(weighting_scheme="recency", divergence_threshold=0.5)
    assert cfg.weighting_scheme == "recency"
    assert cfg.divergence_threshold == 0.5
