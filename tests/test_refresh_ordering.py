"""The top-down refresh must not alter the state that was just evaluated.

The refresh re-seeds cohort adapters FROM the updated population — it is the
starting point for round r+1. It used to run at the END of ``run_round``, so the
caller evaluated round r on state that had already been pulled toward the
population.

The magnitude was not small. At the first sync the population adapter does not
exist, so ``propagate_cohorts_to_population`` returns
``eta_population * mean(cohorts)`` = 0.1 * mean. The refresh then computes
``0.7*cohort + 0.3*(0.1*mean) ≈ 0.73*cohort`` — and the FFN adapter is a RESIDUAL
that is identity at initialisation, so scaling it by 0.73 moves the model ~27 %
back toward the UNADAPTED population model, immediately before the reported
number was taken.

It hit ``hierarchical`` alone: ``top_down_only`` and ``local_finetune`` push the
sync out of reach and never refresh. ``hierarchical`` selecting round 1 in every
completed cell while ``fedavg`` selected round 5 is the visible symptom.
"""
from __future__ import annotations

import pytest
import torch

from cocofl.hierarchical.upward_prop import propagate_cohorts_to_population


ETA_POP = 0.1
BETA = 0.3


def _first_sync_population(cohorts):
    """What the population adapter is after the FIRST sync: eta * mean."""
    return propagate_cohorts_to_population(
        None, cohorts, [1] * len(cohorts), eta_population=ETA_POP, scheme="uniform")


def _refresh(cohort, population, beta=BETA):
    return {k: (1 - beta) * v + beta * population.get(k, torch.zeros_like(v))
            for k, v in cohort.items()}


def test_population_after_first_sync_is_only_eta_of_the_mean():
    a = {"w": torch.ones(4)}
    pop = _first_sync_population([a, a, a])
    assert torch.allclose(pop["w"], torch.full((4,), ETA_POP)), pop["w"]


def test_refresh_attenuates_the_adapter_by_27_percent():
    """The number that matters: how much of the learned adapter survives."""
    a = {"w": torch.ones(4)}
    pop = _first_sync_population([a, a, a])
    out = _refresh(a, pop)
    surviving = float(out["w"][0])
    assert surviving == pytest.approx(0.73, abs=1e-6)
    assert (1 - surviving) == pytest.approx(0.27, abs=1e-6)


def test_attenuation_moves_a_residual_adapter_toward_identity():
    """An FFN adapter is identity at init, so shrinking it toward zero moves the
    model back toward the un-adapted population model — which is precisely the
    baseline it is supposed to be beating."""
    trained = {"w": torch.tensor([2.0, -2.0])}
    pop = _first_sync_population([trained])
    out = _refresh(trained, pop)
    assert out["w"].abs().sum() < trained["w"].abs().sum()


def test_refresh_is_deferred_not_removed():
    """It still happens — at the start of the next round, where it belongs."""
    import inspect

    from cocofl.hierarchical.coordinator import HierarchicalCoordinator

    # `_finish_round` holds step 4 onward and is called at the END of
    # `run_round`, so concatenating them in call order reproduces the round's
    # execution order — which is what this test is really pinning.
    src = (inspect.getsource(HierarchicalCoordinator.run_round)
           + inspect.getsource(HierarchicalCoordinator._finish_round))
    assert "_refresh_pending = True" in src, "sync must schedule a refresh"
    assert "self._refresh_cohort_from_population()" in src, "and it must be applied"
    sched = src.index("_refresh_pending = True")
    applied = src.index("self._refresh_cohort_from_population()")
    assert applied < sched, (
        "the refresh must be APPLIED at the top of the round and SCHEDULED at the "
        "bottom; if it is applied after being scheduled it is still altering the "
        "round that is about to be evaluated"
    )


def test_refresh_flag_starts_clear():
    import inspect

    from cocofl.hierarchical.coordinator import HierarchicalCoordinator

    src = inspect.getsource(HierarchicalCoordinator.__init__)
    assert "_refresh_pending = False" in src


def test_zero_blend_disables_attenuation_entirely():
    a = {"w": torch.ones(4)}
    pop = _first_sync_population([a, a, a])
    out = _refresh(a, pop, beta=0.0)
    assert torch.allclose(out["w"], a["w"]), "beta=0 must be a no-op"
