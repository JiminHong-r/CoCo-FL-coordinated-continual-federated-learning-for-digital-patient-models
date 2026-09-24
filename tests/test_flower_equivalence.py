"""Flower's FedAvg and the single-process FedAvg must produce the same model.

The claim being defended is "moving to a real federated runtime does not change
any reported number". That claim is cheap to assert and easy to get wrong, so it
is checked numerically here rather than argued in prose.

Two things can break it:

1. **Different aggregation.** Flower computes ``sum(n_i * w_i) / sum(n_i)``;
   ``fedavg_states`` computes ``sum((n_i / sum n_j) * w_i)``. Algebraically equal,
   but only if both weight by the SAME per-client sample count.
2. **Different ordering.** Flower exchanges an ordered list of arrays with the
   names stripped. If client and server disagree on the order, tensors are
   averaged against the wrong counterparts — and nothing raises, the numbers are
   simply wrong. This is the more dangerous failure of the two.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cocofl.federated.fedavg import fedavg_states
from cocofl.federated.flower_app import ndarrays_to_state, state_to_ndarrays

flwr = pytest.importorskip("flwr", reason="flwr is an optional extra")
from flwr.server.strategy.aggregate import aggregate  # noqa: E402


def _state(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "down.weight": torch.randn(4, 8, generator=g),
        "down.bias": torch.randn(4, generator=g),
        "up.weight": torch.randn(8, 4, generator=g),
        "classifier.0.weight": torch.randn(3, 8, generator=g),
    }


# --- ordering round-trip: the silent-corruption failure --------------------

def test_state_survives_a_round_trip_through_the_wire_format():
    s = _state(0)
    back = ndarrays_to_state(state_to_ndarrays(s), s)
    assert set(back) == set(s)
    for k in s:
        assert torch.allclose(back[k], s[k]), k


def test_ordering_is_by_sorted_key_so_client_and_server_agree():
    s = _state(0)
    arrays = state_to_ndarrays(s)
    for arr, k in zip(arrays, sorted(s)):
        assert arr.shape == tuple(s[k].shape), k


def test_a_shape_mismatch_raises_instead_of_silently_averaging():
    s = _state(0)
    arrays = state_to_ndarrays(s)
    arrays[0] = np.zeros((99, 99), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        ndarrays_to_state(arrays, s)


def test_wrong_array_count_raises():
    s = _state(0)
    with pytest.raises(ValueError, match="expected"):
        ndarrays_to_state(state_to_ndarrays(s)[:-1], s)


# --- the equivalence that matters ------------------------------------------

@pytest.mark.parametrize("counts", [
    [100, 100, 100],          # equal sizes
    [2855, 134086, 27141],    # the real alpha=0.5 centre sizes
    [1, 1, 100000],           # one dominant centre
])
def test_flower_fedavg_equals_single_process_fedavg(counts):
    states = [_state(i) for i in range(len(counts))]

    ours = fedavg_states(states, [float(c) for c in counts])

    theirs_arrays = aggregate([(state_to_ndarrays(s), c)
                               for s, c in zip(states, counts)])
    theirs = ndarrays_to_state(theirs_arrays, states[0])

    assert set(ours) == set(theirs)
    for k in ours:
        assert torch.allclose(ours[k], theirs[k], atol=1e-6, rtol=1e-5), (
            f"{k}: Flower and the single-process path disagree — the runtime "
            f"would change the reported numbers"
        )


@pytest.mark.parametrize("counts", [
    [100, 100, 100],
    [2855, 134086, 27141],
    [1, 1, 100000],
])
def test_flower_fedavg_equals_the_function_that_produced_THE_RESULTS(counts):
    """`run_standard_fedavg` aggregates with `weighted_state_average`, NOT with
    `fedavg_states`. Testing only the latter would verify a code path that no
    reported number ever went through — the two are duplicate implementations of
    the same formula, and duplicates are exactly how these drift apart."""
    from cocofl.hierarchical.upward_prop import weighted_state_average

    states = [_state(i) for i in range(len(counts))]
    reported = weighted_state_average(states, [float(c) for c in counts])
    flower = ndarrays_to_state(
        aggregate([(state_to_ndarrays(s), c) for s, c in zip(states, counts)]),
        states[0])
    for k in reported:
        assert torch.allclose(reported[k], flower[k], atol=1e-6, rtol=1e-5), k


def test_the_two_in_tree_aggregators_agree_with_each_other():
    """`fedavg_states` and `weighted_state_average` are separate copies of one
    formula. If they ever diverge, `standard_fedavg` and the federated CLI would
    silently disagree about what FedAvg means."""
    from cocofl.hierarchical.upward_prop import weighted_state_average

    states = [_state(i) for i in range(3)]
    w = [2855.0, 134086.0, 27141.0]
    a, b = fedavg_states(states, w), weighted_state_average(states, w)
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-7), k


def test_hierarchical_propagation_is_NOT_fedavg_and_needs_a_custom_strategy():
    """The hierarchy's upward step is an EMA blend, not a weighted average, so
    Flower's built-in FedAvg does not express it. Recording this as a test keeps
    the distinction from being lost: `standard_fedavg` maps onto the stock
    strategy, `hierarchical` requires a custom `Strategy` subclass."""
    from cocofl.hierarchical.upward_prop import (
        propagate_patients_to_cohort, weighted_state_average,
    )

    # Keys are namespaced: a cohort key maps to the patient namespace by suffix,
    # so `cohort_adapter.down.weight` is updated from `patient_adapter.down.weight`.
    # Using bare keys silently matches nothing and returns the cohort unchanged.
    cohort = {"cohort_adapter.down.weight": torch.zeros(4, 8)}
    delta = {"patient_adapter.down.weight": torch.ones(4, 8)}
    blended = propagate_patients_to_cohort(cohort, [delta], [1], eta_cohort=0.3)

    # EMA blend: new = cohort + eta * delta = 0 + 0.3 * 1
    assert torch.allclose(blended["cohort_adapter.down.weight"],
                          torch.full((4, 8), 0.3), atol=1e-6)

    # A weighted average of the same two would give 0.5, not 0.3 — so the
    # upward step is not expressible by the stock FedAvg strategy.
    plain = weighted_state_average(
        [{"x": torch.zeros(4, 8)}, {"x": torch.ones(4, 8)}], [1.0, 1.0])
    assert torch.allclose(plain["x"], torch.full((4, 8), 0.5), atol=1e-6)
    assert not torch.allclose(blended["cohort_adapter.down.weight"], plain["x"]), (
        "if these matched, the upward step would be plain averaging and stock "
        "FedAvg would suffice"
    )


def test_equivalence_holds_for_a_single_client():
    """One client must be a no-op average, not a divide-by-something-else."""
    s = _state(7)
    ours = fedavg_states([s], [42.0])
    theirs = ndarrays_to_state(aggregate([(state_to_ndarrays(s), 42)]), s)
    for k in ours:
        assert torch.allclose(ours[k], s[k], atol=1e-6)
        assert torch.allclose(theirs[k], s[k], atol=1e-6)


def test_weighting_actually_depends_on_the_counts():
    """Guards the test above: if both ignored counts they would agree trivially."""
    a, b = _state(1), _state(2)
    even = fedavg_states([a, b], [1.0, 1.0])
    skew = fedavg_states([a, b], [1.0, 99.0])
    assert not torch.allclose(even["down.weight"], skew["down.weight"]), (
        "weighting has no effect, so the equivalence test proves nothing"
    )


# --- the custom strategy for the hierarchy ---------------------------------

def test_custom_strategy_actually_blends_and_does_not_silently_no_op():
    """The failure this guards against is specific and has already happened
    twice while writing this file: `propagate_*` functions match keys by
    namespace, and given the wrong namespace they return the input UNCHANGED
    without raising. A strategy that no-ops looks like a strategy that works."""
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    from cocofl.federated.flower_app import (
        make_hierarchical_strategy, ndarrays_to_state, state_to_ndarrays,
    )

    Strategy = make_hierarchical_strategy()
    ref = {"cohort_adapter.down.weight": torch.zeros(4, 8)}
    strat = Strategy(reference_state=ref, eta_population=0.1, scheme="volume")

    class _Res:
        def __init__(self, state, n):
            self.parameters = ndarrays_to_parameters(state_to_ndarrays(state))
            self.num_examples = n
            self.metrics = {}

    client_state = {"cohort_adapter.down.weight": torch.ones(4, 8)}
    params, _ = strat.aggregate_fit(1, [(None, _Res(client_state, 100))], [])
    out = ndarrays_to_state(parameters_to_ndarrays(params), ref)

    # EMA: (1-0.1)*0 + 0.1*1 = 0.1
    got = out["cohort_adapter.down.weight"]
    assert torch.allclose(got, torch.full((4, 8), 0.1), atol=1e-6), got[0, 0]
    assert not torch.allclose(got, torch.zeros(4, 8)), (
        "strategy returned the state unchanged — it is a silent no-op"
    )


def test_custom_strategy_differs_from_stock_fedavg():
    """If it matched stock FedAvg there would be no reason to subclass."""
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    from cocofl.federated.flower_app import (
        make_hierarchical_strategy, ndarrays_to_state, state_to_ndarrays,
    )

    Strategy = make_hierarchical_strategy()
    ref = {"cohort_adapter.down.weight": torch.zeros(4, 8)}
    strat = Strategy(reference_state=ref, eta_population=0.1)

    class _Res:
        def __init__(self, state, n):
            self.parameters = ndarrays_to_parameters(state_to_ndarrays(state))
            self.num_examples = n
            self.metrics = {}

    cs = {"cohort_adapter.down.weight": torch.ones(4, 8)}
    params, _ = strat.aggregate_fit(1, [(None, _Res(cs, 100))], [])
    ema = ndarrays_to_state(parameters_to_ndarrays(params), ref)["cohort_adapter.down.weight"]

    stock = ndarrays_to_state(
        aggregate([(state_to_ndarrays(cs), 100)]), ref)["cohort_adapter.down.weight"]

    assert torch.allclose(stock, torch.ones(4, 8), atol=1e-6)
    assert not torch.allclose(ema, stock), (
        "EMA blend and weighted average coincide — stock FedAvg would suffice"
    )


# --- standard_fedavg through Flower must equal the in-process aggregator ------

@pytest.mark.parametrize("counts", [(100, 100, 100), (2855, 134086, 27141), (1, 1, 500)])
def test_flower_fedavg_round_matches_weighted_state_average(counts):
    """`standard_fedavg` is compared directly against `hierarchical` in the
    results table, so both must come from the same runtime. This pins that
    routing it through Flower does not move the number."""
    from cocofl.federated.flower_app import flower_fedavg_round
    from cocofl.hierarchical.upward_prop import weighted_state_average

    server = _state(0)
    locals_ = [_state(1), _state(2), _state(3)]

    # Flower path: each "client" returns its prepared state and sample count.
    train_fns = {
        hid: (lambda st, i=hid: (locals_[i], counts[i])) for hid in range(3)
    }
    got, got_counts = flower_fedavg_round(server, train_fns)

    want = weighted_state_average(locals_, [float(c) for c in counts])

    assert got_counts == {0: counts[0], 1: counts[1], 2: counts[2]}
    for k in want:
        assert torch.allclose(got[k], want[k], atol=1e-6), (
            f"{k} differs between Flower's aggregate and weighted_state_average"
        )


def test_flower_fedavg_round_actually_weights_by_count():
    """A guard: if both aggregators ignored the counts this would still pass
    above, so check the weighting is load-bearing."""
    from cocofl.federated.flower_app import flower_fedavg_round

    server = _state(0)
    a, b = _state(1), _state(2)
    lopsided, _ = flower_fedavg_round(
        server, {0: lambda st: (a, 1000), 1: lambda st: (b, 1)})
    balanced, _ = flower_fedavg_round(
        server, {0: lambda st: (a, 1), 1: lambda st: (b, 1)})
    assert not torch.allclose(lopsided["down.weight"], balanced["down.weight"]), (
        "aggregation ignored the sample counts"
    )


def test_round_trip_preserves_device_not_just_dtype():
    """The wire format is CPU numpy, so a naive rebuild silently moves state to
    CPU and the mismatch only surfaces later, against a live cuda tensor. Pin
    placement here, where the cause is visible."""
    ref = _state(7)
    got = ndarrays_to_state(state_to_ndarrays(ref), ref)
    for k in ref:
        assert got[k].device == ref[k].device, f"{k} changed device on round-trip"
        assert got[k].dtype == ref[k].dtype, f"{k} changed dtype on round-trip"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_round_trip_preserves_cuda_placement():
    ref = {k: v.cuda() for k, v in _state(8).items()}
    got = ndarrays_to_state(state_to_ndarrays(ref), ref)
    for k in ref:
        assert got[k].is_cuda, f"{k} came back on CPU from a cuda reference"
