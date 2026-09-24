"""HierarchicalCoordinator: runs the three-level adaptation loop (WP4 task 5).

One round:

    1. each centre trains its cohort adapter on its own data (medium pace)
    2. each centre trains patient adapters on top of its cohort (fast pace)
    3. each centre folds its patient updates into its cohort adapter (upward step 1)
    4. the cohort states are blended into the population adapter (upward step 2)
    5. cohorts are pulled part-way back towards the population (top-down refresh)

Step 4 runs only when a trigger fires: every ``population_update_freq`` rounds,
after ``population_update_after_n_patients`` patient updates, or when a cohort
drifts more than ``divergence_threshold`` from the population state. Step 5 is
applied at the start of the following round.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.utils.data import DataLoader

from ..models.classifier import TFNDownstreamClassifier
from ..adapters import (
    attach_ffn_adapter, level_state_dict, load_level_state_dict,
)
from .cohort import CohortConfig
from .patient import PatientConfig, train_patient_adapter
from .upward_prop import (
    propagate_cohorts_to_population, propagate_patients_to_cohort,
)


@dataclass
class CoordinatorConfig:
    cohort_cfg: CohortConfig = field(default_factory=CohortConfig)
    patient_cfg: PatientConfig = field(default_factory=PatientConfig)
    patients_per_round: int = 5
    # WP4 task 5 triggers for the population update: every N rounds, or when a
    # cohort has drifted too far from the last population state. Whichever fires
    # first wins.
    population_update_freq: int = 3
    divergence_threshold: float | None = None  # None = frequency-only triggering
    # Trigger after N patient updates. This differs from the round trigger because
    # one round already holds patients_per_round x centres patient updates. None
    # turns it off.
    population_update_after_n_patients: int | None = None
    eta_cohort: float = 0.3
    eta_population: float = 0.1
    refresh_cohort_from_population: bool = True
    # Top-down refresh strength. Each cohort is blended towards the population
    # instead of overwritten, so it keeps what it learned:
    #     cohort <- (1 - refresh_blend) * cohort + refresh_blend * population
    # 1.0 overwrites completely; 0.0 turns the refresh off.
    refresh_blend: float = 0.3
    # Weighting for the two upward steps (WP4 task 3), settable per level so the two
    # can differ. weighting_scheme is used wherever a per-level scheme is unset.
    weighting_scheme: str = "volume"
    patient_weighting_scheme: str | None = None    # patient -> cohort
    cohort_weighting_scheme: str | None = None     # cohort  -> population
    # Half-life, in days, for recency weighting.
    recency_half_life_days: float = 180.0
    # Which patients each round uses. False (default): the same first
    # patients_per_round patients every round. True: a window that moves through the
    # whole list, so more distinct patients contribute over the rounds at the same
    # cost per round. This changes the method, so keep it off when comparing with
    # the reported runs.
    patient_rotation: bool = False
    # Where a patient adapter starts. False (hierarchical): from its centre's cohort
    # state, which is the hierarchy's downward flow. True (local_finetune): from the
    # population model, so no hierarchy is used, matching "per-centre and per-
    # patient fine-tuning without coordination".
    patient_init_from_population: bool = False
    # Run steps 1-3 as Flower clients and step 4 as a Flower strategy. The maths is
    # the same as the in-process loop (tests/test_flower_equivalence.py), but a
    # Flower client does all its own work inside fit(), which changes the order of
    # random draws. Results therefore agree within a small tolerance rather than
    # exactly.
    use_flower: bool = False


class HierarchicalCoordinator:
    """Stateful coordinator that runs the bidirectional adaptation loop."""

    def __init__(self,
                 population_clf: TFNDownstreamClassifier,
                 hospital_train_loaders: dict[int, DataLoader],
                 hospital_patient_streams: dict[int, list[tuple[DataLoader, DataLoader]]],
                 label_fn: Callable,
                 device: torch.device,
                 cfg: CoordinatorConfig | None = None):
        """``hospital_patient_streams[hid]`` is a list of
        ``(train_loader, eval_loader)`` per patient in that hospital."""
        self.population_clf = population_clf
        self.hospital_train_loaders = hospital_train_loaders
        self.hospital_patient_streams = hospital_patient_streams
        self.label_fn = label_fn
        self.device = device
        self.cfg = cfg or CoordinatorConfig()

        # One fresh cohort adapter state per hospital.
        self.cohort_states: dict[int, dict[str, torch.Tensor]] = {}
        for hid in hospital_train_loaders:
            tmpl = copy.deepcopy(population_clf)
            attach_ffn_adapter(tmpl.backbone, bottleneck=self.cfg.cohort_cfg.bottleneck,
                               attr="cohort_adapter")
            self.cohort_states[hid] = level_state_dict(
                tmpl, attr="cohort_adapter",
                include_head=self.cfg.cohort_cfg.train_head,
                head_bias_only=self.cfg.cohort_cfg.train_head_bias_only)
            del tmpl
        # Untouched starting states. The adapter is the identity at initialisation,
        # so these equal the population model; used when patients start from the
        # population (patient_init_from_population).
        self._initial_cohort_states = {
            hid: {k: v.clone() for k, v in st.items()}
            for hid, st in self.cohort_states.items()
        }
        # No population adapter until the first upward step.
        self.population_adapter_state: dict[str, torch.Tensor] | None = None
        self.round_idx = 0
        self.history: list[dict] = []
        # Patient updates since the last population sync, for the "after N patient
        # updates" trigger.
        self.patients_since_sync = 0
        # Set when a population sync happens and used at the start of the next
        # round, so the round just trained is evaluated before any refresh.
        self._refresh_pending = False
        # State for the Flower path: log parts and pre-patient snapshots the clients
        # fill in, plus the clients and strategy, built on first use.
        self._flower_logs: dict = {}
        self._flower_pre_patient: dict = {}
        self._flower_refresh_pending = False
        self._flower_clients = None
        self._flower_strategy = None

        # Replay buffer for the "replay" CL strategy, filled from the cohort
        # training loaders. One buffer is shared by all hospitals so each cohort
        # sees varied samples.
        self.replay_buffer = None
        needs_replay = (
            self.cfg.cohort_cfg.cl_strategy == "replay"
            or self.cfg.patient_cfg.cl_strategy == "replay"
        )
        if needs_replay:
            from ..personalisation.continual import ReplayBufferTFN
            self.replay_buffer = ReplayBufferTFN(
                capacity=self.cfg.cohort_cfg.cl_replay_capacity,
            )
            for loader in self.hospital_train_loaders.values():
                self.replay_buffer.fill_from(loader, self.label_fn)
                if len(self.replay_buffer.items) >= self.replay_buffer.capacity:
                    break

    def _refresh_cohort_from_population(self) -> None:
        """Top-down step: re-init cohorts using population state as starting point."""
        if self.population_adapter_state is None or not self.cfg.refresh_cohort_from_population:
            return
        # Blend rather than overwrite, so each centre keeps what it just learned.
        # Overwriting made the cohort models collapse back to the population model.
        beta = self.cfg.refresh_blend
        if beta <= 0.0:
            return
        for hid in self.cohort_states:
            self._refresh_one_cohort(hid, self.population_adapter_state, beta)

    def _refresh_one_cohort(self, hid, population_state, beta=None) -> None:
        """Blend the population adapter into one centre's cohort state.

        Separate from ``_refresh_cohort_from_population`` because under Flower each
        client does this itself when it receives the population adapter. Same maths,
        done where it would happen in a real deployment.
        """
        if not population_state:
            return
        if beta is None:
            beta = self.cfg.refresh_blend
        if beta <= 0.0:
            return
        for k in self.cohort_states[hid]:
            if k in population_state:
                self.cohort_states[hid][k] = (
                    (1.0 - beta) * self.cohort_states[hid][k]
                    + beta * population_state[k]
                )

    @staticmethod
    def _latest_time_in(loader) -> int | None:
        """Newest prediction time among a loader's windows, in microseconds.

        Recency weighting uses this, so an update's weight depends on how old the
        patient's data is. Returns None when the dataset has no ``times`` or
        ``start_indices``; ``_weights_for`` then warns and falls back.
        """
        subset = getattr(loader, "dataset", None)
        ds = getattr(subset, "dataset", None)
        idxs = getattr(subset, "indices", None)
        if ds is None or idxs is None:
            return None
        starts = getattr(ds, "start_indices", None)
        times = getattr(ds, "times", None)
        if starts is None or times is None:
            return None
        try:
            return max(int(times[starts[i]]) for i in idxs)
        except (TypeError, IndexError, KeyError, ValueError):
            return None

    def _patients_this_round(self, patient_streams: list) -> list:
        """The patient streams to train on this round.

        Without rotation: the first ``patients_per_round`` streams every round. With
        rotation the window moves forward by that many each round and wraps around.
        When the window covers the whole list the two are the same.
        """
        n_per_round = self.cfg.patients_per_round
        n = len(patient_streams)
        if not self.cfg.patient_rotation or n_per_round >= n:
            return patient_streams[:n_per_round]
        start = ((self.round_idx - 1) * n_per_round) % n
        return [patient_streams[(start + k) % n] for k in range(n_per_round)]

    def _cohort_step(self, hid) -> dict:
        """Cohort training for one centre (step 1).

        Kept as its own method so the in-process path and the Flower client run the
        same code instead of two copies that could drift apart.
        """
        loader = self.hospital_train_loaders[hid]
        tmpl = copy.deepcopy(self.population_clf)
        attach_ffn_adapter(tmpl.backbone, bottleneck=self.cfg.cohort_cfg.bottleneck,
                           attr="cohort_adapter")
        # Load the current cohort state into a fresh copy.
        load_level_state_dict(tmpl, self.cohort_states[hid], attr="cohort_adapter")
        # Train one local round on top of it.
        for p in tmpl.backbone.cohort_adapter.parameters():
            p.requires_grad_(True)
        for p in tmpl.classifier.parameters():
            p.requires_grad_(self.cfg.cohort_cfg.train_head)
        if (self.cfg.cohort_cfg.train_head_bias_only
                and not self.cfg.cohort_cfg.train_head):
            # Intercepts only (see CohortConfig.train_head_bias_only).
            from ..adapters.ffn import head_bias_parameters
            for _n, p in head_bias_parameters(tmpl):
                p.requires_grad_(True)
        # With cl_strategy == "distill" the frozen population model is the teacher,
        # which keeps the cohort adapter close to population knowledge (WP4 task 4).
        from .cohort import _train_one_cohort_step
        teacher = self.population_clf if self.cfg.cohort_cfg.cl_strategy == "distill" else None
        n_seen, avg_loss = _train_one_cohort_step(
            tmpl, loader, self.label_fn, self.cfg.cohort_cfg, self.device,
            teacher=teacher, replay_buffer=self.replay_buffer,
        )
        self.cohort_states[hid] = level_state_dict(
            tmpl, attr="cohort_adapter",
            include_head=self.cfg.cohort_cfg.train_head,
            head_bias_only=self.cfg.cohort_cfg.train_head_bias_only)
        del tmpl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {f"cohort_{hid}_loss": avg_loss, f"cohort_{hid}_n_seen": n_seen}

    def _patient_step(self, hid, pre_state: dict) -> dict:
        """Patient training and patient-to-cohort propagation for one centre (steps 2-3).

        ``pre_state`` is the centre's cohort state after the cohort step and before
        any patient feedback; the size of the patient update is measured against it
        (WP5 axis 4).
        """
        patient_streams = self.hospital_patient_streams[hid]
        cohort_state = self.cohort_states[hid]
        log: dict = {}
        deltas, counts, losses, times = [], [], [], []
        for (_pid, p_train_loader, _p_val_loader, _p_eval_loader) in self._patients_this_round(patient_streams):
            _patient_base = (self._initial_cohort_states[hid]
                             if self.cfg.patient_init_from_population
                             else cohort_state)
            _, _, delta, p_loss = train_patient_adapter(
                self.population_clf, _patient_base, p_train_loader, self.label_fn,
                self.cfg.patient_cfg, self.device,
                cohort_bottleneck=self.cfg.cohort_cfg.bottleneck,
                replay_buffer=self.replay_buffer,
            )
            deltas.append(delta)
            counts.append(len(p_train_loader.dataset))
            # Per-patient task loss; uncertainty weighting uses 1 / loss.
            losses.append(p_loss)
            # How recent this patient's data is (WP4 task 3, recency weighting).
            times.append(self._latest_time_in(p_train_loader))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Newest data this centre contributed, for cohort-to-population recency.
        _cohort_times = [t for t in times if t is not None]
        log[f"cohort_{hid}_latest_time"] = (
            max(_cohort_times) if _cohort_times
            else self._latest_time_in(self.hospital_train_loaders.get(hid))
        )
        if deltas:
            self.cohort_states[hid] = propagate_patients_to_cohort(
                cohort_state, deltas, counts,
                eta_cohort=self.cfg.eta_cohort,
                scheme=(self.cfg.patient_weighting_scheme
                        or self.cfg.weighting_scheme),
                losses=losses,
                times=times,
                half_life_days=self.cfg.recency_half_life_days,
            )
        self.patients_since_sync += len(deltas)
        log[f"cohort_{hid}_n_patients"] = len(deltas)
        # Size (L2 norm) of the change patient feedback made to the cohort.
        delta_norm = 0.0
        for k, v_new in self.cohort_states[hid].items():
            if k in pre_state:
                delta_norm += float((v_new - pre_state[k]).pow(2).sum())
        log[f"cohort_{hid}_patient_delta_norm"] = delta_norm ** 0.5
        return log

    def run_round(self) -> dict:
        """One round: cohort, then patients, then upward steps, then maybe a population update."""
        self.round_idx += 1
        round_log = {"round": self.round_idx}

        # Apply the refresh from the previous round's population sync here, so it
        # seeds this round's training instead of changing the state already reported
        # for the previous round.
        if getattr(self, "_refresh_pending", False):
            self._refresh_cohort_from_population()
            self._refresh_pending = False
            round_log["refreshed_from_population"] = True

        # Steps 1-3: per-centre work, in-process or through Flower
        if self.cfg.use_flower:
            cohort_pre_patient = self._flower_centre_rounds(round_log)
            round_log["cohort_state_pre_patient"] = cohort_pre_patient
            return self._finish_round(round_log)

        # Step 1: cohort training, per centre
        for hid in self.hospital_train_loaders:
            round_log.update(self._cohort_step(hid))
        # Snapshot each cohort before patient feedback, so later changes can be
        # credited to patient feedback (WP5 axis 4).
        cohort_pre_patient = {
            hid: {k: v.clone() for k, v in cs.items()}
            for hid, cs in self.cohort_states.items()
        }
        round_log["cohort_state_pre_patient"] = cohort_pre_patient

        # Steps 2-3: patient training and upward propagation, per centre
        for hid in self.hospital_patient_streams:
            round_log.update(self._patient_step(hid, cohort_pre_patient[hid]))

        return self._finish_round(round_log)

    # ------------------------------------------------------------------ Flower
    def _flower_centre_rounds(self, round_log: dict) -> dict:
        """Steps 1-3 for every centre, run as Flower clients.

        Each client runs its own cohort step and patient loop inside ``fit()``, as a
        real federated client would; this is why the order differs from the in-
        process path. The clients call the same ``_cohort_step`` and
        ``_patient_step`` as above.
        """
        from ..federated.flower_app import (
            make_hierarchical_client_class, state_to_ndarrays,
        )
        if getattr(self, "_flower_clients", None) is None:
            Client = make_hierarchical_client_class()
            self._flower_clients = {
                hid: Client(self, hid) for hid in self.hospital_train_loaders
            }
        # The server sends the population adapter; when a refresh is due, each
        # client blends it in on arrival (the downward flow).
        pop = self.population_adapter_state or {}
        params = state_to_ndarrays(pop) if pop else []
        refresh = bool(getattr(self, "_flower_refresh_pending", False))
        self._flower_refresh_pending = False
        self._flower_pre_patient = {}
        for hid, client in self._flower_clients.items():
            _arrays, _n, _metrics = client.fit(
                params, {"refresh": refresh, "round": self.round_idx})
            round_log.update(self._flower_logs.pop(hid, {}))
        return self._flower_pre_patient

    def _flower_aggregate(self, cohort_states, cohort_counts,
                          cohort_losses, cohort_times):
        """Step 4 through a Flower strategy.

        ``HierarchicalEMAStrategy`` subclasses Flower's ``FedAvg`` and replaces the
        plain average with the cohort-to-population blend, which stock FedAvg cannot
        express. Checked in tests/test_flower_equivalence.py.
        """
        from ..federated.flower_app import (
            make_hierarchical_strategy, state_to_ndarrays, ndarrays_to_state,
        )
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

        reference = cohort_states[0]
        if getattr(self, "_flower_strategy", None) is None:
            Strategy = make_hierarchical_strategy()
            self._flower_strategy = Strategy(
                reference_state=reference,
                eta_population=self.cfg.eta_population,
                scheme=(self.cfg.cohort_weighting_scheme
                        or self.cfg.weighting_scheme),
            )
            self._flower_strategy._current = (
                {k: v.clone() for k, v in self.population_adapter_state.items()}
                if self.population_adapter_state else
                {k: torch.zeros_like(v) for k, v in reference.items()}
            )
        strat = self._flower_strategy
        # Log that the upward step ran and by which route, so a run's log shows it
        # directly.
        print(f"    cohort->population via FLOWER strategy "
              f"(HierarchicalEMAStrategy, eta_population={self.cfg.eta_population}, "
              f"scheme={self.cfg.cohort_weighting_scheme or self.cfg.weighting_scheme}, "
              f"round={self.round_idx}, n_cohorts={len(cohort_states)})")
        strat.round_losses = cohort_losses
        strat.round_times = cohort_times
        strat.half_life_days = self.cfg.recency_half_life_days

        class _Res:
            def __init__(self, state, n):
                self.parameters = ndarrays_to_parameters(state_to_ndarrays(state))
                self.num_examples = int(n)
                self.metrics = {}

        results = [(None, _Res(s, n)) for s, n in zip(cohort_states, cohort_counts)]
        params, _ = strat.aggregate_fit(self.round_idx, results, [])
        return ndarrays_to_state(parameters_to_ndarrays(params), reference)

    def _finish_round(self, round_log: dict) -> dict:
        """Step 4 onward — shared by the in-process and Flower paths."""
        # Step 4: cohort -> population (slow). It fires on round frequency, after N
        # patient updates, or on divergence; whichever comes first.
        freq_trigger = self.round_idx % self.cfg.population_update_freq == 0
        n_patient_trigger = False
        if self.cfg.population_update_after_n_patients is not None:
            n_patient_trigger = (
                self.patients_since_sync >= self.cfg.population_update_after_n_patients
            )
        round_log["patients_since_sync"] = self.patients_since_sync
        diverge_trigger = False
        max_diverge = None
        if self.cfg.divergence_threshold is not None:
            # L2 distance from each cohort to the last population state. If any
            # cohort is past the threshold, update the population now. Before the
            # first update the population adapter is all zeros (the identity), so
            # divergence is measured from zero; it does not wait for a frequency
            # update to happen first.
            pop_state = self.population_adapter_state or {}
            divs = []
            for cs in self.cohort_states.values():
                d_sq = 0.0
                for k, v in cs.items():
                    ref = pop_state.get(k)
                    d_sq += float(((v - ref) if ref is not None else v).pow(2).sum())
                divs.append(d_sq ** 0.5)
            max_diverge = max(divs)
            diverge_trigger = max_diverge >= self.cfg.divergence_threshold
            round_log["max_cohort_divergence"] = max_diverge
        if freq_trigger or diverge_trigger or n_patient_trigger:
            cohort_states = list(self.cohort_states.values())
            cohort_counts = [round_log.get(f"cohort_{h}_n_seen", 1)
                             for h in self.cohort_states]
            # Per-cohort task loss from this round, in the same order as
            # cohort_states. Uncertainty weighting uses 1 / loss, so a centre whose
            # adapter fits its data well counts more.
            cohort_losses = [
                round_log.get(f"cohort_{h}_loss") if
                round_log.get(f"cohort_{h}_loss") is not None else 1.0
                for h in self.cohort_states
            ]
            # Recency at this level: the newest data any of the cohort's patients
            # contributed, so a centre without recent data counts less. None (with a
            # warning) if unavailable.
            cohort_times = [round_log.get(f"cohort_{h}_latest_time")
                            for h in self.cohort_states]
            if self.cfg.use_flower:
                self.population_adapter_state = self._flower_aggregate(
                    cohort_states, cohort_counts, cohort_losses, cohort_times)
            else:
                print(f"    cohort->population via in-process propagation "
                      f"(eta_population={self.cfg.eta_population}, "
                      f"round={self.round_idx}, n_cohorts={len(cohort_states)})")
                self.population_adapter_state = propagate_cohorts_to_population(
                    self.population_adapter_state, cohort_states, cohort_counts,
                    eta_population=self.cfg.eta_population,
                    scheme=(self.cfg.cohort_weighting_scheme
                            or self.cfg.weighting_scheme),
                    losses=cohort_losses,
                    times=cohort_times,
                    half_life_days=self.cfg.recency_half_life_days,
                )
            round_log["population_updated"] = True
            fired = [name for name, on in (("frequency", freq_trigger),
                                            ("n_patients", n_patient_trigger),
                                            ("divergence", diverge_trigger)) if on]
            round_log["update_trigger"] = "+".join(fired) if fired else "none"
            # Reset the patient-update count only when a sync actually happens.
            self.patients_since_sync = 0

            # Step 5: top-down refresh, deferred to the next round. The refresh is
            # the starting point for round r+1. Applying it now would change the
            # state that is about to be evaluated for round r. At the first sync
            # that shrinks the cohort adapter by about 27 %, pulling hierarchical
            # back towards the un-adapted model right before it is measured.
            if self.cfg.refresh_cohort_from_population:
                if self.cfg.use_flower:
                    # Each client applies it when the next broadcast arrives.
                    self._flower_refresh_pending = True
                else:
                    self._refresh_pending = True

        self.history.append(round_log)
        return round_log
