"""Flower's FedAvg and the in-process FedAvg must give the same model.

Moving to a real federated runtime must not change any reported number. Two things
could break that:

1. Different aggregation. Flower computes ``sum(n_i * w_i) / sum(n_i)`` and
``weighted_state_average`` computes ``sum((n_i / sum n_j) * w_i)``. These are equal
only if both use the same per-client sample counts. 2. Different ordering. Flower
sends an ordered list of arrays without names. If client and server disagree on the
order, the wrong tensors are averaged and nothing raises. This is the more dangerous
of the two.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cocofl.hierarchical.upward_prop import weighted_state_average
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


# Ordering round trip (the silent-corruption case)

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


# The equivalence that matters

@pytest.mark.parametrize("counts", [
    [100, 100, 100],          # equal sizes
    [2855, 134086, 27141],    # the real alpha=0.5 centre sizes
    [1, 1, 100000],           # one dominant centre
])
def test_flower_fedavg_equals_single_process_fedavg(counts):
    states = [_state(i) for i in range(len(counts))]

    ours = weighted_state_average(states, [float(c) for c in counts])

    theirs_arrays = aggregate([(state_to_ndarrays(s), c)
                               for s, c in zip(states, counts)])
    theirs = ndarrays_to_state(theirs_arrays, states[0])

    assert set(ours) == set(theirs)
    for k in ours:
        assert torch.allclose(ours[k], theirs[k], atol=1e-6, rtol=1e-5), (
            f"{k}: Flower and the single-process path disagree — the runtime "
            f"would change the reported numbers"
        )


def test_hierarchical_propagation_is_NOT_fedavg_and_needs_a_custom_strategy():
    """The hierarchy's upward step is a blend, not a weighted average, so it needs a custom Flower strategy; standard_fedavg uses the stock one."""
    from cocofl.hierarchical.upward_prop import (
        propagate_patients_to_cohort, weighted_state_average,
    )

    # Keys are namespaced: cohort_adapter.down.weight is updated from
    # patient_adapter.down.weight by suffix. Bare keys would match nothing and leave
    # the cohort unchanged.
    cohort = {"cohort_adapter.down.weight": torch.zeros(4, 8)}
    delta = {"patient_adapter.down.weight": torch.ones(4, 8)}
    blended = propagate_patients_to_cohort(cohort, [delta], [1], eta_cohort=0.3)

    # Blend: new = cohort + eta * delta = 0 + 0.3 * 1
    assert torch.allclose(blended["cohort_adapter.down.weight"],
                          torch.full((4, 8), 0.3), atol=1e-6)

    # A weighted average of the same two gives 0.5, not 0.3, so stock FedAvg cannot
    # express the step.
    plain = weighted_state_average(
        [{"x": torch.zeros(4, 8)}, {"x": torch.ones(4, 8)}], [1.0, 1.0])
    assert torch.allclose(plain["x"], torch.full((4, 8), 0.5), atol=1e-6)
    assert not torch.allclose(blended["cohort_adapter.down.weight"], plain["x"]), (
        "if these matched, the upward step would be plain averaging and stock "
        "FedAvg would suffice"
    )


def test_equivalence_holds_for_a_single_client():
    """With one client the average must return that client's state unchanged."""
    s = _state(7)
    ours = weighted_state_average([s], [42.0])
    theirs = ndarrays_to_state(aggregate([(state_to_ndarrays(s), 42)]), s)
    for k in ours:
        assert torch.allclose(ours[k], s[k], atol=1e-6)
        assert torch.allclose(theirs[k], s[k], atol=1e-6)


def test_weighting_actually_depends_on_the_counts():
    """Guard for the tests above: if both ignored the counts they would agree trivially."""
    a, b = _state(1), _state(2)
    even = weighted_state_average([a, b], [1.0, 1.0])
    skew = weighted_state_average([a, b], [1.0, 99.0])
    assert not torch.allclose(even["down.weight"], skew["down.weight"]), (
        "weighting has no effect, so the equivalence test proves nothing"
    )


# The custom strategy for the hierarchy

def test_custom_strategy_actually_blends_and_does_not_silently_no_op():
    """The strategy must really blend. The propagate_* functions match keys by namespace and return the input unchanged, without error, when given the wrong namespace."""
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

    # Blend: (1 - 0.1) * 0 + 0.1 * 1 = 0.1
    got = out["cohort_adapter.down.weight"]
    assert torch.allclose(got, torch.full((4, 8), 0.1), atol=1e-6), got[0, 0]
    assert not torch.allclose(got, torch.zeros(4, 8)), (
        "strategy returned the state unchanged — it is a silent no-op"
    )


def test_custom_strategy_differs_from_stock_fedavg():
    """If it matched stock FedAvg there would be no reason to subclass it."""
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


# standard_fedavg through Flower must equal the in-process aggregator

@pytest.mark.parametrize("counts", [(100, 100, 100), (2855, 134086, 27141), (1, 1, 500)])
def test_flower_fedavg_round_matches_weighted_state_average(counts):
    """standard_fedavg is compared directly with hierarchical, so routing it through Flower must not change its number."""
    from cocofl.federated.flower_app import flower_fedavg_round
    from cocofl.hierarchical.upward_prop import weighted_state_average

    server = _state(0)
    locals_ = [_state(1), _state(2), _state(3)]

    # Flower path: each client returns its prepared state and sample count.
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
    """Guard: check that the counts really change the result, or the test above could pass trivially."""
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
    """The wire format is CPU numpy; rebuilding must restore the original device, not only the dtype, or the error appears later, far from its cause."""
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
