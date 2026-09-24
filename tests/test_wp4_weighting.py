"""WP4 task 3/5 knobs must actually reach the coordinator and do something.

Two failures this pins, both of which produced results that *looked* fine:

* ``weighting_scheme`` and ``divergence_threshold`` were never read by
  ``run_evaluation.py``, so every reported run used volume + frequency-only no
  matter what was passed on the command line.
* ``uncertainty`` weighting was fed a hardcoded loss of ``1.0`` for every
  patient, so its inverse-loss weights were all equal — it was ``uniform`` under
  a different name, and an ablation comparing the two would have shown a
  spurious "no difference".
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
    """Position decay is NOT recency — every update in a round is equally recent.

    Kept only for callers that cannot supply timestamps, and it must warn: a
    silent fallback here is precisely the defect that made the task-3 `recency`
    arm measure arbitrary list ordering. Time-based behaviour is pinned in
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
    """The cohort->population step must supply losses, not just patient->cohort.

    The 2026-08-08 fix wired real losses into ``propagate_patients_to_cohort``
    only. ``propagate_cohorts_to_population`` still passed ``scheme`` with no
    ``losses``, so every ``weighting_scheme=uncertainty`` run died on the first
    population sync with "uncertainty weighting requires losses" — the whole
    task-3 arm produced no results at all while the ablation table listed it as
    still running.
    """
    from cocofl.hierarchical.upward_prop import propagate_cohorts_to_population

    cohorts = [{"a": torch.tensor([1.0, 0.0])}, {"a": torch.tensor([0.0, 1.0])}]
    out = propagate_cohorts_to_population(
        None, cohorts, [10, 10], eta_population=1.0,
        scheme="uncertainty", losses=[0.1, 1.0],
    )
    # Cohort 0 has the lower loss, so it dominates the aggregate.
    assert out["a"][0] > out["a"][1]

    with pytest.raises(ValueError, match="requires losses"):
        propagate_cohorts_to_population(
            None, cohorts, [10, 10], scheme="uncertainty",
        )


def test_coordinator_passes_losses_to_the_population_step():
    """Pins the call site, not just the callee — the bug was a missing kwarg."""
    import inspect
    import re

    from cocofl.hierarchical.coordinator import HierarchicalCoordinator

    # Step 4 lives in `_finish_round`, which `run_round` tail-calls.
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
    """This is exactly what the hardcoded 1.0 produced — the bug, pinned.

    If a future change reintroduces a constant loss, the uncertainty arm of the
    task-3 ablation silently becomes a duplicate of the uniform arm.
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
    """The coordinator needs a real per-patient loss for uncertainty weighting."""
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
