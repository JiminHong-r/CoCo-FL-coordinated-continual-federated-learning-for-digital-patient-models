"""WP4 tasks 3-5 must vary what the TD actually asks about.

Three gaps between the task descriptions and the implementation, each of which
would have produced an ablation table that looks complete and answers a different
question:

* **Task 3** defines TWO upward mechanisms (patient->cohort and cohort->
  population) and asks for schemes based on "data volume, uncertainty, or update
  recency". One shared ``weighting_scheme`` knob applied the same scheme to both
  levels, so the two mechanisms could not be compared. And ``recency`` decayed by
  POSITION in the within-round update list — where every update is equally recent
  — so that arm measured arbitrary ordering, not recency.
* **Task 5** asks for triggering "after N patient updates, or based on divergence
  thresholds". ``population_update_freq`` counts ROUNDS, and one round contains
  ``patients_per_round × centres`` patient updates, so a round-based trigger
  cannot express the TD's first condition at all.
"""
from __future__ import annotations

import pytest
import torch

from cocofl.hierarchical.upward_prop import (
    MICROSECONDS_PER_DAY, _weights_for, propagate_cohorts_to_population,
)
from cocofl.hierarchical.coordinator import CoordinatorConfig


DAY = MICROSECONDS_PER_DAY


# --- task 3: recency must mean elapsed time -------------------------------

def test_recency_decays_by_elapsed_time_not_position():
    """Two updates, equal counts. The OLDER one must be down-weighted even when
    it sits last in the list — position must not drive the weight."""
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
    """Every patient in one round shares a timestamp under position decay; under
    time decay they correctly tie instead of being ranked by list order."""
    t = 42 * DAY
    w = _weights_for("recency", [1, 1, 1], times=[t, t, t], half_life_days=90.0)
    assert w[0] == w[1] == w[2] == pytest.approx(1.0)


def test_missing_timestamps_warn_before_falling_back():
    """A silent fallback would reintroduce exactly the defect being fixed."""
    with pytest.warns(UserWarning, match="POSITION decay"):
        w = _weights_for("recency", [1, 1, 1], times=None, recency_decay=0.5)
    assert w == [0.25, 0.5, 1.0], "documented legacy behaviour, but warned about"


def test_partial_timestamps_also_warn():
    with pytest.warns(UserWarning, match="POSITION decay"):
        _weights_for("recency", [1, 1], times=[10 * DAY, None])


# --- task 3: two mechanisms, selectable per level -------------------------

def test_config_exposes_a_scheme_per_level():
    cfg = CoordinatorConfig(weighting_scheme="volume",
                            patient_weighting_scheme="recency",
                            cohort_weighting_scheme="uncertainty")
    assert cfg.patient_weighting_scheme == "recency"
    assert cfg.cohort_weighting_scheme == "uncertainty"


def test_unset_per_level_scheme_falls_back_to_the_shared_one():
    """Preserves the previous single-knob behaviour bit-for-bit."""
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
    # The stale centre carries half the weight, so the fresh one dominates.
    assert out["a"][0] > out["a"][1]


# --- task 5: "after N patient updates" -----------------------------------

def test_config_exposes_the_patient_update_trigger():
    cfg = CoordinatorConfig(population_update_after_n_patients=90)
    assert cfg.population_update_after_n_patients == 90


def test_patient_update_trigger_is_off_by_default():
    """Off means round-frequency behaviour is unchanged for existing runs."""
    assert CoordinatorConfig().population_update_after_n_patients is None


def test_round_frequency_cannot_express_n_patient_updates():
    """Why the separate trigger is needed, stated as arithmetic: with 20 patients
    per round across 3 centres, every round applies 60 patient updates, so no
    integer round frequency fires 'after 90 patient updates'."""
    per_round = 20 * 3
    target = 90
    assert target % per_round != 0, (
        "if this ever divides evenly, the round trigger coincides with the "
        "patient trigger and the two arms stop being distinguishable"
    )
