"""Run the federated steps through Flower's client/server runtime.

Each centre is a Flower client that only exchanges parameters with a server. The
maths is the same as the in-process loop: Flower's ``FedAvg`` computes

    w_global = sum_i (n_i * w_i) / sum_i n_i

and ``upward_prop.weighted_state_average`` computes

    w_global = sum_i ((n_i / sum_j n_j) * w_i)

which is the same number with the division done in a different place.
``tests/test_flower_equivalence.py`` checks this numerically.

What travels: only the adapter state (plus the classifier head when ``train_head``
is on), the same 66,112 parameters per level the communication analysis counts. The
frozen backbone never moves.

Key order matters: Flower sends an ordered list of arrays without names, so client
and server must use the same order, or the wrong tensors get averaged together with
no error. Every conversion here uses ``sorted(state)``, and the round trip is
tested.
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

    ``reference`` supplies the names the wire format drops. Shapes are checked,
    because a mismatch here would quietly average unrelated tensors.
    """
    keys = sorted(reference)
    if len(arrays) != len(keys):
        raise ValueError(
            f"expected {len(keys)} arrays for keys {keys}, got {len(arrays)}"
        )
    out: dict[str, torch.Tensor] = OrderedDict()
    for k, a in zip(keys, arrays):
        ref = reference[k]
        # Restore the device as well as the dtype. The wire format is CPU numpy, and
        # CPU tensors would only fail later, at the first operation with a CUDA
        # tensor.
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

    ``standard_fedavg`` is a plain weighted average, which Flower's built-in
    ``FedAvg`` computes exactly. The hierarchy instead blends:

        new = (1 - eta) * old + eta * aggregated

    with weights selectable per level (volume, uniform, recency, uncertainty; WP4
    task 3). Stock FedAvg cannot express that, so the strategy is subclassed. (The
    fl-ight ``carlos`` branch's ``FedXGBLLRStrategy`` follows the same pattern.)

    ``local_finetune`` and ``population_only`` need no strategy: the first trains
    each centre with no coordination and the second does not train.
    """
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    from ..hierarchical.upward_prop import propagate_cohorts_to_population

    class HierarchicalEMAStrategy(fl.server.strategy.FedAvg):
        """FedAvg with the average replaced by the cohort-to-population blend.

        It uses ``propagate_cohorts_to_population``, which takes absolute states
        with the same keys on server and clients, as a Flower client returns them.
        ``propagate_patients_to_cohort`` would be wrong here: it renames
        ``patient_adapter.*`` keys to ``cohort_adapter.*``, so given cohort keys it
        matches nothing and silently returns the state unchanged.
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

    Besides the cohort step, the client runs the patient level: each patient adapter
    starts from the cohort state, adapts, and the accepted updates are folded back
    into the cohort before ``fit`` returns.

    The patient level stays local on purpose: patients are records inside one
    institution, not network peers. The only real network boundary is cohort <->
    population, and that is what Flower carries.

    The client calls the coordinator's own ``_cohort_step`` and ``_patient_step``
    instead of reimplementing them, so the two paths cannot drift apart.
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
            # centre's state. The client does it because in a real deployment the
            # client receives the broadcast.
            if parameters is not None and len(parameters) and config.get("refresh"):
                pop = ndarrays_to_state(parameters, coord.cohort_states[hid])
                coord._refresh_one_cohort(hid, pop)

            log = dict(coord._cohort_step(hid))
            # Snapshot after cohort training and before patient feedback, for the
            # axis 4 update size.
            pre = {k: v.clone() for k, v in coord.cohort_states[hid].items()}
            coord._flower_pre_patient[hid] = pre
            if hid in coord.hospital_patient_streams:
                log.update(coord._patient_step(hid, pre))
            coord._flower_logs[hid] = log

            # Flower's FedAvg weights by this count, so it must be the same n_seen
            # the in-process path uses.
            n_seen = int(log.get(f"cohort_{hid}_n_seen", 1) or 1)
            return (state_to_ndarrays(coord.cohort_states[hid]),
                    max(n_seen, 1),
                    {"hospital": int(hid), "n_seen": n_seen})

        def evaluate(self, parameters, config):
            # The harness scores every method centrally on the same windows, so
            # clients report nothing here.
            return 0.0, 0, {}

    return HierarchicalClient


def make_callable_client_class():
    """A Flower client that hands local training to a supplied function.

    The client handles only the Flower side (wire format, ordering, sample counts);
    the caller decides what "train locally" means. ``standard_fedavg`` passes its
    cohort training step (with the distillation teacher when ``use_cl`` is on).
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
            # Flower's FedAvg weights by this count, so it must match the in-process
            # count or the two runtimes would weight centres differently.
            return (state_to_ndarrays(new_state), int(max(n_seen, 1)),
                    {"hospital": int(self.hid), "n_seen": int(n_seen)})

        def evaluate(self, parameters, config):
            return 0.0, 0, {}

    return CallableClient


def flower_fedavg_round(server_state, train_fns):
    """One FedAvg round through Flower: clients train, the stock strategy averages.

    ``train_fns`` maps hospital id -> ``fn(state) -> (new_state, n_seen)``.

    It uses Flower's own ``aggregate`` (``sum(n_i * w_i) / sum(n_i)``) rather than
    an in-tree helper, so ``standard_fedavg`` is aggregated by the framework just
    like the hierarchy. tests/test_flower_equivalence.py checks the result.
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
