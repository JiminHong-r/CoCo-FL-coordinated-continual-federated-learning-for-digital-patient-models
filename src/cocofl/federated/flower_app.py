"""Run the WP4 adapter FedAvg through Flower's client/server runtime.

WHY THIS EXISTS
---------------
``fedavg.py`` simulates federation in one process: a loop over hospitals, then a
weighted average. That is sufficient for the science — and it is what produced
every reported number — but it does not demonstrate that the method runs under a
real federated-learning runtime, where each centre is a separate client that only
ever exchanges parameters with a server.

This module supplies that demonstration WITHOUT changing any result. Flower's
built-in ``FedAvg`` strategy aggregates with

    w_global = sum_i (n_i * w_i) / sum_i n_i          (flwr.server.strategy.aggregate)

and ``fedavg.fedavg_states`` computes

    w_global = sum_i ((n_i / sum_j n_j) * w_i)

which is the same quantity, differing only in the order of the divide. The
equivalence is asserted numerically in ``tests/test_flower_equivalence.py`` rather
than argued, because "the runtime does not change the result" is exactly the kind
of claim that should be checked rather than assumed.

WHAT TRAVELS
------------
Only the adapter state (plus the classifier head when ``train_head`` is on) — the
same 66,112 parameters per level the communication analysis counts. The frozen
backbone never moves, so the Flower payload matches the reported cost.

KEY ORDERING IS LOAD-BEARING
----------------------------
Flower exchanges an ORDERED list of arrays with no names attached. Client and
server must therefore agree on the order, and a mismatch would silently average
the wrong tensors together rather than raise. Every conversion here goes through
``sorted(state)``, and the round-trip is tested.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch



def state_to_ndarrays(state: dict[str, torch.Tensor]) -> list[np.ndarray]:
    """State dict -> ordered arrays, keyed by ``sorted(state)``."""
    return [state[k].detach().cpu().numpy() for k in sorted(state)]


def ndarrays_to_state(arrays: list[np.ndarray],
                      reference: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Ordered arrays -> state dict, using ``reference``'s keys and dtypes.

    ``reference`` supplies the names the wire format drops. Shapes are checked
    because a silent mismatch here would average unrelated tensors.
    """
    keys = sorted(reference)
    if len(arrays) != len(keys):
        raise ValueError(
            f"expected {len(keys)} arrays for keys {keys}, got {len(arrays)}"
        )
    out: dict[str, torch.Tensor] = OrderedDict()
    for k, a in zip(keys, arrays):
        ref = reference[k]
        # DEVICE, not just dtype. The wire format is CPU numpy by construction,
        # so rebuilding without ``device=`` yields CPU tensors while the rest of
        # the pipeline lives on cuda — and the failure surfaces later, at the
        # first arithmetic against a live state ("Expected all tensors to be on
        # the same device"), far from the round trip that caused it. The
        # reference supplies names, dtype AND placement.
        t = torch.as_tensor(a, dtype=ref.dtype, device=ref.device)
        if tuple(t.shape) != tuple(ref.shape):
            raise ValueError(
                f"shape mismatch for {k}: got {tuple(t.shape)}, expected "
                f"{tuple(ref.shape)} — client and server disagree on layout"
            )
        out[k] = t
    return out


def make_hierarchical_strategy():
    """Flower strategy for the hierarchy's upward step.

    WHY A CUSTOM STRATEGY IS NEEDED
    -------------------------------
    Only ONE of the five WP5 methods is FedAvg. `standard_fedavg` aggregates by
    weighted average, which Flower's built-in `FedAvg` computes exactly (see
    tests/test_flower_equivalence.py). The hierarchy does something different:

        new_cohort = cohort + eta_cohort * aggregated_delta

    an EMA blend, where the aggregation weights are themselves selectable per
    level (volume / uniform / recency / uncertainty, TD WP4 task 3). Stock FedAvg
    cannot express that — with eta=0.3 onto a zero cohort it would return 0.5
    where the blend returns 0.3 — so "run the hierarchy through Flower" means
    subclassing the strategy, not reusing it.

    This mirrors the pattern in the fl-ight `carlos` branch, whose
    `FedXGBLLRStrategy` likewise subclasses `FedAvg` to replace the aggregation
    while keeping Flower's client orchestration.

    `local_finetune` and `population_only` need no strategy at all: the first
    trains per centre with no coordination, the second does not train.
    """
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    from ..hierarchical.upward_prop import propagate_cohorts_to_population

    class HierarchicalEMAStrategy(fl.server.strategy.FedAvg):
        """Replaces FedAvg's average with the cohort->population EMA blend.

        ``propagate_cohorts_to_population`` is the correct primitive here, not
        ``propagate_patients_to_cohort``: it consumes **absolute** states with
        keys shared between server and clients, which is what a Flower client
        returns. The patient-level function instead maps keys across namespaces
        (``patient_adapter.*`` -> ``cohort_adapter.*``) and, given
        cohort-namespaced input, would match nothing and return the state
        unchanged — silently, with no error.
        """

        def __init__(self, *args, reference_state, eta_population=0.1,
                     scheme="volume", **kwargs):
            super().__init__(*args, **kwargs)
            self.reference_state = reference_state
            self.eta_population = eta_population
            self.scheme = scheme
            self._current = {k: v.clone() for k, v in reference_state.items()}

        def aggregate_fit(self, server_round, results, failures):
            if not results:
                return None, {}
            states, counts = [], []
            for _client, fit_res in results:
                states.append(ndarrays_to_state(
                    parameters_to_ndarrays(fit_res.parameters),
                    self.reference_state))
                counts.append(int(fit_res.num_examples))
            self._current = propagate_cohorts_to_population(
                self._current, states, counts,
                eta_population=self.eta_population, scheme=self.scheme,
                losses=getattr(self, "round_losses", None),
                times=getattr(self, "round_times", None),
                half_life_days=getattr(self, "half_life_days", 180.0))
            return ndarrays_to_parameters(state_to_ndarrays(self._current)), {}

    return HierarchicalEMAStrategy


def make_hierarchical_client_class():
    """One Flower client per centre, for the full three-level hierarchy.

    WHY THIS IS SEPARATE FROM ``make_client_class``
    ----------------------------------------------
    That one is the ``standard_fedavg`` baseline: its ``fit`` trains a cohort
    adapter and stops. The hierarchy's client must also run the PATIENT level —
    instantiate each patient adapter from the cohort state, adapt it, and
    propagate the accepted deltas back into the cohort — before returning.

    THE PATIENT LEVEL IS LOCAL BY CONSTRUCTION, NOT BY COMPROMISE
    ------------------------------------------------------------
    Patients are records inside one institution; they are not network peers and
    never contact a server. The only boundary that is genuinely a network
    boundary is cohort <-> population, and that is the one Flower carries. A
    three-level hierarchy over a two-tier runtime is the correct topology.

    IT CALLS THE COORDINATOR'S OWN STEPS
    ------------------------------------
    ``_cohort_step`` and ``_patient_step`` are invoked rather than reimplemented.
    Re-deriving a per-centre step inside the client is exactly how the same logic
    ended up in two copies that drifted apart once already.
    """
    import flwr as fl

    class HierarchicalClient(fl.client.NumPyClient):
        def __init__(self, coord, hid):
            self.coord = coord
            self.hid = hid

        def get_parameters(self, config):  # noqa: D102
            return state_to_ndarrays(self.coord.cohort_states[self.hid])

        def fit(self, parameters, config):
            """One federated round for this centre: refresh, cohort, patients."""
            coord, hid = self.coord, self.hid
            # Downward flow: blend the broadcast population adapter into this
            # centre's state. Done here because on a real deployment the client
            # is what receives the broadcast.
            if parameters is not None and len(parameters) and config.get("refresh"):
                pop = ndarrays_to_state(parameters, coord.cohort_states[hid])
                coord._refresh_one_cohort(hid, pop)

            log = dict(coord._cohort_step(hid))
            # Snapshot AFTER cohort training and BEFORE patient feedback — the
            # reference the axis-4 delta norm is measured against.
            pre = {k: v.clone() for k, v in coord.cohort_states[hid].items()}
            coord._flower_pre_patient[hid] = pre
            if hid in coord.hospital_patient_streams:
                log.update(coord._patient_step(hid, pre))
            coord._flower_logs[hid] = log

            # Flower's FedAvg weights by this count, so it must be the SAME
            # ``n_seen`` the in-process path puts into ``cohort_counts``.
            n_seen = int(log.get(f"cohort_{hid}_n_seen", 1) or 1)
            return (state_to_ndarrays(coord.cohort_states[hid]),
                    max(n_seen, 1),
                    {"hospital": int(hid), "n_seen": n_seen})

        def evaluate(self, parameters, config):
            # The harness scores every method on identical windows centrally, so
            # clients report nothing; a placeholder keeps the strategy happy.
            return 0.0, 0, {}

    return HierarchicalClient


def make_callable_client_class():
    """A Flower client that delegates local training to a supplied callable.

    WHY GENERIC RATHER THAN ONE CLIENT PER METHOD
    ---------------------------------------------
    The two federated methods do not train the same way. ``standard_fedavg``
    trains a shared cohort adapter with ``_train_one_cohort_step`` (and the
    distillation teacher when ``use_cl`` is on); the WP4 path trains via
    ``_local_train_one_hospital``; the hierarchy additionally runs a patient
    loop. Hard-coding any of them into a client forces the others to be
    reimplemented next to it — the duplication that has already cost this project
    a defect, when per-centre state-reinstall logic drifted apart in two copies.

    So this client owns the Flower contract (wire format, ordering, sample
    counts) and nothing else. The caller owns what "train locally" means.
    """
    import flwr as fl

    class CallableClient(fl.client.NumPyClient):
        def __init__(self, hid, train_fn, reference_state):
            self.hid = hid
            self.train_fn = train_fn
            self.reference = reference_state

        def get_parameters(self, config):  # noqa: D102
            return state_to_ndarrays(self.reference)

        def fit(self, parameters, config):
            state = ndarrays_to_state(parameters, self.reference)
            new_state, n_seen = self.train_fn(state)
            # Flower's FedAvg weights by this count, so it must match the count
            # the single-process path uses or the two runtimes would weight
            # differently and diverge despite identical arithmetic.
            return (state_to_ndarrays(new_state), int(max(n_seen, 1)),
                    {"hospital": int(self.hid), "n_seen": int(n_seen)})

        def evaluate(self, parameters, config):
            return 0.0, 0, {}

    return CallableClient


def flower_fedavg_round(server_state, train_fns):
    """One FedAvg round through Flower: clients train, stock strategy averages.

    ``train_fns`` maps hospital id -> ``fn(state) -> (new_state, n_seen)``.

    Uses Flower's OWN ``aggregate`` (``sum(n_i * w_i) / sum(n_i)``) rather than
    an in-tree helper, so ``standard_fedavg`` is aggregated by the framework
    exactly as the hierarchy's strategy is. That equality is asserted in
    tests/test_flower_equivalence.py rather than assumed.
    """
    from flwr.server.strategy.aggregate import aggregate

    Client = make_callable_client_class()
    params = state_to_ndarrays(server_state)
    results, counts = [], {}
    for hid, fn in train_fns.items():
        client = Client(hid, fn, server_state)
        arrays, n, metrics = client.fit(params, {})
        results.append((arrays, n))
        counts[hid] = int(metrics.get("n_seen", n))
    return ndarrays_to_state(aggregate(results), server_state), counts
