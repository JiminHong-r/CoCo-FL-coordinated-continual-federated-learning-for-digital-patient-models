"""WP4 tasks 3-5 must vary what the TD actually asks about.

* Task 3 defines two upward steps (patient -> cohort, cohort -> population) with
weighting by data volume, uncertainty or update recency. The scheme must be settable
per step, and recency must mean elapsed time: decaying by list position is
meaningless when every update in a round is equally recent. * Task 5 asks for a
trigger "after N patient updates". Counting rounds cannot express that, because one
round holds patients_per_round x centres updates.
"""
from __future__ import annotations

import pytest
import torch

from cocofl.hierarchical.upward_prop import (
    MICROSECONDS_PER_DAY, _weights_for, propagate_cohorts_to_population,
)
from cocofl.hierarchical.coordinator import CoordinatorConfig


DAY = MICROSECONDS_PER_DAY


# Task 3: recency means elapsed time

def test_recency_decays_by_elapsed_time_not_position():
    """With equal counts, the older update gets less weight even when it comes last in the list."""
    now = 1_000 * DAY
    times = [now, now - 180 * DAY]          # newest FIRST, oldest LAST
    w = _weights_for("recency", [1, 1], times=times, half_life_days=180.0)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.5), "one half-life older -> half the weight"


def test_recency_half_life_is_in_days():
    now = 500 * DAY
    w = _weights_for("recency", [1, 1, 1],
                     times=[now, now - 30 * DAY, now - 60 * DAY],
                     half_life_days=30.0)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.5)
    assert w[2] == pytest.approx(0.25)


def test_equally_recent_updates_get_equal_weight():
    """Updates with the same timestamp get equal weight instead of being ranked by list order."""
    t = 42 * DAY
    w = _weights_for("recency", [1, 1, 1], times=[t, t, t], half_life_days=90.0)
    assert w[0] == w[1] == w[2] == pytest.approx(1.0)


def test_missing_timestamps_warn_before_falling_back():
    """Falling back to position decay must warn."""
    with pytest.warns(UserWarning, match="POSITION decay"):
        w = _weights_for("recency", [1, 1, 1], times=None, recency_decay=0.5)
    assert w == [0.25, 0.5, 1.0], "documented legacy behaviour, but warned about"


def test_partial_timestamps_also_warn():
    with pytest.warns(UserWarning, match="POSITION decay"):
        _weights_for("recency", [1, 1], times=[10 * DAY, None])


# Task 3: two steps, each with its own scheme

def test_config_exposes_a_scheme_per_level():
    cfg = CoordinatorConfig(weighting_scheme="volume",
                            patient_weighting_scheme="recency",
                            cohort_weighting_scheme="uncertainty")
    assert cfg.patient_weighting_scheme == "recency"
    assert cfg.cohort_weighting_scheme == "uncertainty"


def test_unset_per_level_scheme_falls_back_to_the_shared_one():
    """With no per-level scheme set, both levels use the shared one, as before."""
    cfg = CoordinatorConfig(weighting_scheme="uniform")
    assert cfg.patient_weighting_scheme is None
    assert cfg.cohort_weighting_scheme is None
    assert (cfg.patient_weighting_scheme or cfg.weighting_scheme) == "uniform"
    assert (cfg.cohort_weighting_scheme or cfg.weighting_scheme) == "uniform"


def test_cohort_to_population_accepts_times():
    now = 10 * DAY
    states = [{"a": torch.tensor([1.0, 0.0])}, {"a": torch.tensor([0.0, 1.0])}]
    out = propagate_cohorts_to_population(
        None, states, [5, 5], eta_population=1.0, scheme="recency",
        times=[now, now - 180 * DAY], half_life_days=180.0,
    )
    # The older centre gets half the weight, so the newer one counts more.
    assert out["a"][0] > out["a"][1]


# Task 5: "after N patient updates"

def test_config_exposes_the_patient_update_trigger():
    cfg = CoordinatorConfig(population_update_after_n_patients=90)
    assert cfg.population_update_after_n_patients == 90


def test_patient_update_trigger_is_off_by_default():
    """Off by default, so existing runs keep the round-frequency behaviour."""
    assert CoordinatorConfig().population_update_after_n_patients is None


def test_round_frequency_cannot_express_n_patient_updates():
    """Why a separate trigger is needed: at 20 patients per round across 3 centres every round applies 60 updates, so no round frequency fires after exactly 90."""
    per_round = 20 * 3
    target = 90
    assert target % per_round != 0, (
        "if this ever divides evenly, the round trigger coincides with the "
        "patient trigger and the two arms stop being distinguishable"
    )
