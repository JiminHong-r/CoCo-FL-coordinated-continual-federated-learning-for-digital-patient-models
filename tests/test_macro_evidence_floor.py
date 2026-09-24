"""MACRO must mean the same thing at every α, for every task and method.

Three defects this pins, all of which produced numbers that looked comparable
and were not:

* **No evidence floor.** A Dirichlet draw at low α leaves centres with a handful
  of patients — at α=0.1 hospital 2 holds 69 — and their AUROCs are coin flips.
  ``readmission`` at α=0.1 scored 0.4 / 0.5 / 0.3 / 0.8 across four methods on
  one such centre. Averaged with equal weight, a 10-pair centre moved MACRO as
  much as a 100,000-pair one, so MACRO could not be compared across α.
* **Sizes only from population_only.** ``n_eval_windows`` was read off
  ``pop_res``, but the WP4 ablations run ``hierarchical`` alone, so their CSVs
  carried no sizes and no confidence intervals — nothing to compare against the
  main runs with.
* **No record of which round was selected.** When selection lands on round 0 the
  method IS the population baseline, so every ablation knob is a no-op and all
  arms report bit-identical numbers. Three CL strategies "agreeing" to 16
  decimal places at α=0.1 was that, not a finding.
"""
from __future__ import annotations

import pytest

from cocofl.evaluation.run_evaluation import (
    _eval_sizes, _macro_over, _reportable_hospitals,
)
from cocofl.evaluation.result import MethodResult


# --- evidence floor -------------------------------------------------------

def test_centre_with_too_few_positives_is_excluded():
    ns = {0: 100_000, 1: 500}
    n_pos = {0: 2_000, 1: 3}          # hospital 1: 3 positives
    keep = _reportable_hospitals([0, 1], ns, n_pos, min_pos=10, min_neg=10)
    assert keep == [0]


def test_centre_with_too_few_negatives_is_excluded():
    ns = {0: 100_000, 1: 105}
    n_pos = {0: 2_000, 1: 100}        # hospital 1: only 5 negatives
    keep = _reportable_hospitals([0, 1], ns, n_pos, min_pos=10, min_neg=10)
    assert keep == [0]


def test_the_alpha_0_1_case_that_motivated_this():
    """69-patient centre against a 134,000-patient one."""
    ns = {0: 246_000, 2: 104}
    n_pos = {0: 5_100, 2: 2}
    keep = _reportable_hospitals([0, 2], ns, n_pos, min_pos=10, min_neg=10)
    assert keep == [0], "the coin-flip centre must not enter MACRO"


def test_unknown_sizes_are_kept():
    """Absent evidence of a problem, dropping a cell would hide a real result."""
    keep = _reportable_hospitals([0, 1], {}, {}, min_pos=10, min_neg=10)
    assert keep == [0, 1]


def test_floor_of_zero_reproduces_the_old_unguarded_behaviour():
    ns, n_pos = {0: 100_000, 1: 4}, {0: 2_000, 1: 1}
    keep = _reportable_hospitals([0, 1], ns, n_pos, min_pos=0, min_neg=0)
    assert keep == [0, 1], "min_eval_pos=0 must restore the legacy averages"


# --- restricted macro -----------------------------------------------------

def test_macro_ignores_hospitals_outside_the_floor():
    aucs = {0: 0.90, 1: 0.30}         # hospital 1 is the coin flip
    assert _macro_over(aucs, [0]) == pytest.approx(0.90)
    assert _macro_over(aucs, [0, 1]) == pytest.approx(0.60)


def test_macro_skips_none_within_the_kept_set():
    aucs = {0: 0.80, 1: None, 2: 0.90}
    assert _macro_over(aucs, [0, 1, 2]) == pytest.approx(0.85)


def test_macro_of_nothing_is_none():
    assert _macro_over({0: 0.9}, []) is None


# --- sizes travel with every method --------------------------------------

def test_sizes_fall_back_to_whichever_method_ran():
    """The ablations run `hierarchical` alone — sizes must still be found."""
    empty = MethodResult()
    hier = MethodResult(ns={0: 1_234}, n_pos={0: 56})
    ns, n_pos = _eval_sizes(empty, hier)
    assert ns == {0: 1_234} and n_pos == {0: 56}


def test_sizes_prefer_the_first_populated_result():
    a = MethodResult(ns={0: 1}, n_pos={0: 1})
    b = MethodResult(ns={0: 2}, n_pos={0: 2})
    assert _eval_sizes(a, b)[0] == {0: 1}


def test_sizes_of_no_results_are_empty_not_an_error():
    assert _eval_sizes(MethodResult()) == ({}, {})


# --- selection provenance -------------------------------------------------

def test_method_result_records_selected_round_and_movement():
    r = MethodResult(selected_round=0, selected_delta_norm=0.0)
    assert r.selected_round == 0
    assert r.selected_delta_norm == 0.0


def test_selected_round_defaults_to_none_not_zero():
    """None means "not recorded"; 0 is a real, meaningful answer (no adaptation
    was selected). Conflating them would make an unrecorded run look like a
    round-0 fallback."""
    assert MethodResult().selected_round is None


def test_state_delta_norm_is_zero_only_for_an_unchanged_state():
    import torch

    from cocofl.evaluation.baselines import state_delta_norm

    init = {0: {"a": torch.zeros(2)}}
    assert state_delta_norm({0: {"a": torch.zeros(2)}}, init) == 0.0
    assert state_delta_norm({0: {"a": torch.tensor([3.0, 4.0])}}, init) == 5.0
