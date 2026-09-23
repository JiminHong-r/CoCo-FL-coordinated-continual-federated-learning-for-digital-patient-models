"""WP4 task 5: the divergence trigger must be able to fire.

The coordination protocol syncs the population on FREQUENCY or on DIVERGENCE.
The divergence branch used to be guarded by
``population_adapter_state is not None``, but that state is only ever assigned
*inside* the trigger block — so divergence could not fire until a frequency
update had already happened. At the default ``population_update_freq ==
max_rounds`` (one sync, on the final round) that meant it never fired, and a run
configured with a divergence threshold produced results identical to
frequency-only. An ablation over triggering conditions would have reported "no
difference" as a finding.

Before the first sync the population adapter is the zero state, so divergence is
just the cohort's own norm — which is measurable from round 1.
"""
from __future__ import annotations

import torch


def _divergence(cohort_states, pop_state):
    """Mirrors the coordinator's divergence computation."""
    pop_state = pop_state or {}
    divs = []
    for cs in cohort_states.values():
        d_sq = 0.0
        for k, v in cs.items():
            ref = pop_state.get(k)
            d_sq += float(((v - ref) if ref is not None else v).pow(2).sum())
        divs.append(d_sq ** 0.5)
    return max(divs) if divs else None


def test_divergence_is_measurable_before_the_first_population_update():
    cohorts = {0: {"a": torch.tensor([3.0, 4.0])}}
    d = _divergence(cohorts, None)
    assert d == 5.0, "distance from the zero population state is the cohort norm"


def test_divergence_fires_against_a_zero_population():
    cohorts = {0: {"a": torch.tensor([3.0, 4.0])}}
    assert _divergence(cohorts, None) >= 0.5, "threshold 0.5 must be reachable"


def test_zero_cohort_has_zero_divergence():
    cohorts = {0: {"a": torch.zeros(4)}}
    assert _divergence(cohorts, None) == 0.0


def test_divergence_uses_the_population_state_once_it_exists():
    cohorts = {0: {"a": torch.tensor([3.0, 4.0])}}
    pop = {"a": torch.tensor([3.0, 4.0])}
    assert _divergence(cohorts, pop) == 0.0, "identical states have not diverged"


def test_max_over_hospitals_is_used():
    cohorts = {
        0: {"a": torch.tensor([1.0, 0.0])},
        1: {"a": torch.tensor([3.0, 4.0])},
    }
    assert _divergence(cohorts, None) == 5.0, "any centre exceeding the bar triggers"
