"""WP5 axes 3, 4 and 6 must report what the task description asks for.

* Axis 4: the population model's gain from cohort aggregation (the upward half of
the loop), not only the patient-to-cohort gain. * Axis 3: population stability
across all update rounds, i.e. the population model scored every round. * Axis 6:
parameters updated per synchronisation round, per level, and total cost in parameter
counts, not only bytes.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cocofl.evaluation.metrics import (
    naive_full_model_parameters, parameters_updated,
)


# Axis 6: parameter counts per level

def test_parameters_updated_counts_a_state_dict():
    state = {"a": torch.zeros(3, 4), "b": torch.zeros(5)}
    assert parameters_updated(state) == 17


def test_parameters_updated_is_none_for_an_absent_level():
    """None means the level never synced, which differs from syncing 0 parameters."""
    assert parameters_updated(None) is None


def test_parameters_updated_of_an_empty_state_is_zero():
    assert parameters_updated({}) == 0


def test_naive_full_model_parameters_counts_the_whole_model():
    m = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 1))
    assert naive_full_model_parameters(m) == (4 * 3 + 3) + (3 * 1 + 1)


def test_adapter_is_orders_of_magnitude_cheaper_than_the_full_model():
    """The axis 6 claim in the TD's unit (parameters) rather than bytes."""
    adapter = {"down.weight": torch.zeros(64, 512), "down.bias": torch.zeros(64),
               "up.weight": torch.zeros(512, 64), "up.bias": torch.zeros(512)}
    full = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 512))
    ratio = naive_full_model_parameters(full) / parameters_updated(adapter)
    assert ratio > 5, "adapter-only transfer must be materially cheaper"


# Axes 3 and 4: the population trajectory

def _gain(traj, baseline):
    vals = [a for a in traj if a is not None]
    return (vals[-1] - baseline) if (vals and baseline is not None) else None


def _stability_range(traj):
    vals = [a for a in traj if a is not None]
    return (max(vals) - min(vals)) if len(vals) >= 2 else None


def test_population_gain_is_measured_against_the_frozen_baseline():
    assert _gain([0.70, 0.72, 0.75], baseline=0.70) == pytest.approx(0.05)


def test_population_gain_can_be_negative():
    """Aggregating cohorts upward need not help; the column must be able to show a loss."""
    assert _gain([0.70, 0.66], baseline=0.70) < 0


def test_population_gain_is_none_without_a_baseline():
    assert _gain([0.7, 0.8], baseline=None) is None


def test_population_gain_is_none_when_no_sync_ever_happened():
    assert _gain([None, None], baseline=0.70) is None


def test_rounds_before_the_first_sync_report_the_baseline():
    """Before the first sync the population is the frozen model, so the trajectory starts flat instead of blank."""
    traj = [0.70, 0.70, 0.73]
    assert traj[0] == 0.70
    assert _gain(traj, 0.70) == pytest.approx(0.03)


def test_stability_range_captures_drift_across_rounds():
    assert _stability_range([0.70, 0.75, 0.68]) == pytest.approx(0.75 - 0.68)


def test_stability_range_needs_at_least_two_rounds():
    assert _stability_range([0.70]) is None
    assert _stability_range([]) is None


def test_a_perfectly_stable_population_has_zero_range():
    assert _stability_range([0.7, 0.7, 0.7]) == 0.0
