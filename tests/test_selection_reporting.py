"""A results row must never contradict the model it describes.

The bias-only arm reported ``hierarchical`` at ``selected_round=0`` alongside
``|Δ| = 7.13`` — "no adaptation happened" printed next to "the weights moved a
lot". Both came from the same defect: two call sites swapped in the per-centre
states without updating the numbers reported beside them.

The val AUROC was the damaging half. Under per-centre selection the installed
ensemble is each centre's own best round, so its macro is ``mean(per_hosp_best)``
and is >= the best global macro by construction. Reporting the global figure
UNDERSTATES the model actually being tested, and validation AUROC is the exact
quantity the "does adaptation beat population_only?" comparison rests on.
"""
from __future__ import annotations

import pytest

from cocofl.evaluation.baselines import resolve_selection


G_STATES = {0: {"w": "global0"}, 1: {"w": "global1"}}
P_STATES = {0: {"w": "perhosp0"}, 1: {"w": "perhosp1"}}


def _resolve(on: bool, per_hosp_best=None, global_val=0.80):
    return resolve_selection(
        per_hospital_selection=on,
        global_states=G_STATES, global_val=global_val, global_round=2,
        per_hosp_state=P_STATES,
        per_hosp_best=per_hosp_best if per_hosp_best is not None
        else {0: 0.90, 1: 0.70},
        per_hosp_round={0: 3, 1: 5},
    )


# --- off: nothing changes -------------------------------------------------

def test_global_selection_is_passed_through_untouched():
    s = _resolve(on=False)
    assert s.states is G_STATES
    assert s.val_auroc == 0.80
    assert s.round == 2
    assert s.per_hospital_round is None


# --- on: states, val and round must all agree ------------------------------

def test_per_centre_selection_installs_the_per_centre_states():
    assert _resolve(on=True).states is P_STATES


def test_val_auroc_describes_the_installed_ensemble_not_the_global_choice():
    s = _resolve(on=True)
    assert s.val_auroc == pytest.approx(0.80)          # mean(0.90, 0.70)
    # The point of the test: it is recomputed, not inherited. Make the global
    # figure distinct so equality cannot pass by coincidence.
    s2 = _resolve(on=True, global_val=0.55)
    assert s2.val_auroc == pytest.approx(0.80), (
        "val_auroc was inherited from the global choice — it must describe the "
        "per-centre ensemble that was actually installed"
    )


def test_per_centre_val_is_never_below_the_global_choice():
    """Selecting per centre maximises over rounds independently, so its macro
    dominates the best single global round. A violation means the two are being
    computed from different things."""
    s = _resolve(on=True, per_hosp_best={0: 0.91, 1: 0.85}, global_val=0.87)
    assert s.val_auroc >= 0.87


def test_selected_round_is_none_because_centres_chose_different_rounds():
    s = _resolve(on=True)
    assert s.round is None, (
        "a single round label is not defined under per-centre selection; "
        "reporting one produces rows that contradict their own |Δ|"
    )


def test_the_rounds_actually_used_are_still_reported():
    assert _resolve(on=True).per_hospital_round == {0: 3, 1: 5}


def test_per_hospital_round_is_a_copy_not_a_live_reference():
    src = {0: 3, 1: 5}
    s = resolve_selection(
        per_hospital_selection=True, global_states=G_STATES, global_val=0.8,
        global_round=2, per_hosp_state=P_STATES, per_hosp_best={0: 0.9, 1: 0.7},
        per_hosp_round=src,
    )
    src[0] = 99
    assert s.per_hospital_round == {0: 3, 1: 5}


def test_all_none_val_does_not_raise():
    s = _resolve(on=True, per_hosp_best={0: None, 1: None})
    assert s.val_auroc is None


# --- the CSV formatter ----------------------------------------------------

def test_per_hospital_round_column_formatting():
    from cocofl.evaluation.run_evaluation import _fmt_per_hosp_round

    class _R:
        extra = {"selected_round_per_hospital": {1: 5, 0: 3}}

    assert _fmt_per_hosp_round(_R()) == "0:3 1:5"       # sorted by hospital


@pytest.mark.parametrize("extra", [None, {}, {"selected_round_per_hospital": None}])
def test_per_hospital_round_column_is_blank_under_global_selection(extra):
    from cocofl.evaluation.run_evaluation import _fmt_per_hosp_round

    class _R:
        pass

    _R.extra = extra
    assert _fmt_per_hosp_round(_R()) is None


# --- one implementation, not two -----------------------------------------

def test_neither_runner_reinstalls_per_centre_states_by_hand():
    """Both call sites must go through resolve_selection. An inline
    ``best_states = per_hosp_state`` is how the two copies drifted apart."""
    import inspect

    from cocofl.evaluation import baselines, run_evaluation

    for mod in (baselines, run_evaluation):
        src = inspect.getsource(mod)
        body = src[src.index("def resolve_selection") + 1:] \
            if "def resolve_selection" in src else src
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not (stripped.endswith("= _per_hosp_state")
                        or stripped.endswith("= per_hosp_state")), (
                f"{mod.__name__} assigns the per-centre states directly; "
                f"route it through resolve_selection so val/round stay consistent"
            )
