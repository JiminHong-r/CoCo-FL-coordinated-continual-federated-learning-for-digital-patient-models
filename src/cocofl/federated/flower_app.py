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
from typing import Callable

import numpy as np
import torch

from .fedavg import FedRoundConfig, _local_train_one_hospital, load_federated_state


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
        t = torch.as_tensor(a, dtype=ref.dtype)
        if tuple(t.shape) != tuple(ref.shape):
            raise ValueError(
                f"shape mismatch for {k}: got {tuple(t.shape)}, expected "
                f"{tuple(ref.shape)} — client and server disagree on layout"
            )
        out[k] = t
    return out


def make_client_class():
    """Build the NumPyClient subclass, importing flwr lazily.

    Lazy so that importing ``cocofl.federated`` does not require flwr: it is an
    optional extra (``pip install -e '.[federated]'``) and the single-process
    path in ``fedavg.py`` must keep working without it.
    """
    import flwr as fl

    class AdapterClient(fl.client.NumPyClient):
        """One simulated centre. Trains the adapter locally, returns it."""

        def __init__(self, hid, template, loader, round_cfg, device,
                     label_fn, init_state):
            self.hid = hid
            self.template = template
            self.loader = loader
            self.round_cfg = round_cfg
            self.device = device
            self.label_fn = label_fn
            self.state = {k: v.clone() for k, v in init_state.items()}

        def get_parameters(self, config):  # noqa: D102
            return state_to_ndarrays(self.state)

        def fit(self, parameters, config):
            """Local training for one round.

            The returned sample count is what Flower's FedAvg weights by, so it
            must be the same ``n_seen`` the single-process path uses — otherwise
            the two would weight differently and the runtimes would diverge.
            """
            self.state = ndarrays_to_state(parameters, self.state)
            local = _clone_with_state(self.template, self.state, self.device)
            new_state, n_seen = _local_train_one_hospital(
                local, self.loader, self.label_fn, self.round_cfg, self.device)
            self.state = new_state
            del local
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return (state_to_ndarrays(new_state), int(max(n_seen, 1)),
                    {"hospital": int(self.hid), "n_seen": int(n_seen)})

        def evaluate(self, parameters, config):
            # Evaluation is done centrally by the harness on identical windows,
            # so clients report nothing here; returning a placeholder keeps the
            # strategy from treating this as an error.
            return 0.0, 0, {}

    return AdapterClient


def _clone_with_state(template, state, device):
    import copy
    clf = copy.deepcopy(template)
    load_federated_state(clf, state)
    return clf.to(device)


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
                eta_population=self.eta_population, scheme=self.scheme)
            return ndarrays_to_parameters(state_to_ndarrays(self._current)), {}

    return HierarchicalEMAStrategy
