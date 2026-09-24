"""MACRO must mean the same thing at every alpha, for every task and method.

* Evidence floor: at low alpha a centre can be tiny (69 patients at alpha=0.1) and
its AUROC is close to a coin flip, yet with equal weight it moves MACRO as much as a
large centre. * Sizes from any method: runs with only ``hierarchical`` (the WP4
ablations) must still report eval-set sizes and confidence intervals. * Record the
selected round: if selection picks round 0 the method equals the population
baseline, and all arms give identical numbers.
"""
from __future__ import annotations

import pytest

from cocofl.evaluation.run_evaluation import (
    _eval_sizes, _macro_over, _reportable_hospitals,
)
from cocofl.evaluation.result import MethodResult


# Evidence floor

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
    """A 69-patient centre next to a 134,000-patient one."""
    ns = {0: 246_000, 2: 104}
    n_pos = {0: 5_100, 2: 2}
    keep = _reportable_hospitals([0, 2], ns, n_pos, min_pos=10, min_neg=10)
    assert keep == [0], "the coin-flip centre must not enter MACRO"


def test_unknown_sizes_are_kept():
    """Hospitals with unknown sizes are kept, not silently dropped."""
    keep = _reportable_hospitals([0, 1], {}, {}, min_pos=10, min_neg=10)
    assert keep == [0, 1]


def test_floor_of_zero_reproduces_the_old_unguarded_behaviour():
    ns, n_pos = {0: 100_000, 1: 4}, {0: 2_000, 1: 1}
    keep = _reportable_hospitals([0, 1], ns, n_pos, min_pos=0, min_neg=0)
    assert keep == [0, 1], "min_eval_pos=0 must restore the legacy averages"


# MACRO over the kept hospitals only

def test_macro_ignores_hospitals_outside_the_floor():
    aucs = {0: 0.90, 1: 0.30}         # hospital 1 is the coin flip
    assert _macro_over(aucs, [0]) == pytest.approx(0.90)
    assert _macro_over(aucs, [0, 1]) == pytest.approx(0.60)


def test_macro_skips_none_within_the_kept_set():
    aucs = {0: 0.80, 1: None, 2: 0.90}
    assert _macro_over(aucs, [0, 1, 2]) == pytest.approx(0.85)


def test_macro_of_nothing_is_none():
    assert _macro_over({0: 0.9}, []) is None


# Sizes come with every method

def test_sizes_fall_back_to_whichever_method_ran():
    """The ablations run only hierarchical; the sizes must still be found."""
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


# Which round was selected

def test_method_result_records_selected_round_and_movement():
    r = MethodResult(selected_round=0, selected_delta_norm=0.0)
    assert r.selected_round == 0
    assert r.selected_delta_norm == 0.0


def test_selected_round_defaults_to_none_not_zero():
    """None means "not recorded"; 0 is a real answer (no adaptation selected). The two must not be confused."""
    assert MethodResult().selected_round is None


def test_state_delta_norm_is_zero_only_for_an_unchanged_state():
    import torch

    from cocofl.evaluation.baselines import state_delta_norm

    init = {0: {"a": torch.zeros(2)}}
    assert state_delta_norm({0: {"a": torch.zeros(2)}}, init) == 0.0
    assert state_delta_norm({0: {"a": torch.tensor([3.0, 4.0])}}, init) == 5.0
