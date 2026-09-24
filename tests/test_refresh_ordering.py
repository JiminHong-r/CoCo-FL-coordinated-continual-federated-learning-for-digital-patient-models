"""The top-down refresh must not change the state that is about to be evaluated.

The refresh sets the starting point for the next round. Applying it at the end of a
round would change round r's state before it is scored. At the first sync the
population adapter is ``eta_population * mean(cohorts)`` = 0.1 * mean, so the
refresh gives ``0.7 * cohort + 0.3 * (0.1 * mean)``, about ``0.73 * cohort``. The
adapter is a residual that starts at the identity, so that pulls the model about 27
% back towards the un-adapted population model. Only ``hierarchical`` is affected,
since the other methods never sync.
"""
from __future__ import annotations

import pytest
import torch

from cocofl.hierarchical.upward_prop import propagate_cohorts_to_population


ETA_POP = 0.1
BETA = 0.3


def _first_sync_population(cohorts):
    """The population adapter after the first sync: eta * mean."""
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
    """How much of the learned adapter survives the refresh."""
    a = {"w": torch.ones(4)}
    pop = _first_sync_population([a, a, a])
    out = _refresh(a, pop)
    surviving = float(out["w"][0])
    assert surviving == pytest.approx(0.73, abs=1e-6)
    assert (1 - surviving) == pytest.approx(0.27, abs=1e-6)


def test_attenuation_moves_a_residual_adapter_toward_identity():
    """Shrinking an identity-initialised adapter moves the model back towards the un-adapted population model, the baseline it should beat."""
    trained = {"w": torch.tensor([2.0, -2.0])}
    pop = _first_sync_population([trained])
    out = _refresh(trained, pop)
    assert out["w"].abs().sum() < trained["w"].abs().sum()


def test_refresh_is_deferred_not_removed():
    """The refresh still happens, at the start of the next round."""
    import inspect

    from cocofl.hierarchical.coordinator import HierarchicalCoordinator

    # _finish_round (step 4 onward) is called at the end of run_round, so joining
    # their sources in call order gives the round's execution order.
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
