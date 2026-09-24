"""HierarchicalCoordinator: orchestrates the bidirectional 3-level loop (Spec WP4 task 5).

Round structure (one "super-round"):

  1. For each cohort: train cohort adapter on its data (medium pace)
  2. For each cohort: train patient adapters on top of cohort (fast pace)
  3. Per cohort: aggregate patient deltas -> update cohort adapter (upward step 1)
  4. Across cohorts: aggregate cohort states -> EMA-update population adapter (upward step 2)
  5. Optional: re-seed cohort adapters from updated population (top-down refresh)

The coordination protocol (Spec WP4 task 5) is controlled by:
  - ``patients_per_round`` : how many patients per cohort to adapt per round
  - ``population_update_freq``: do step 4 every N rounds (slow)
  - ``divergence_threshold`` (optional): trigger sync when cohort drifts too far
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
    # Spec WP4 task 5: trigger upward population update by either a fixed
    # frequency (every N rounds) OR a divergence threshold (cohort drifted
    # too far from the last broadcast). The first condition met wins.
    population_update_freq: int = 3
    divergence_threshold: float | None = None  # None = frequency-only triggering
    # Spec WP4 task 5 asks for triggering "after N patient updates, OR based on
    # divergence thresholds". ``population_update_freq`` counts ROUNDS, which is
    # not the same condition: a round contains ``patients_per_round`` × centres
    # patient updates, so a round-based trigger cannot express "after N patient
    # updates" at all. This counts actual patient updates since the last sync.
    # None = disabled (round frequency and/or divergence still apply).
    population_update_after_n_patients: int | None = None
    eta_cohort: float = 0.3
    eta_population: float = 0.1
    refresh_cohort_from_population: bool = True
    # Top-down refresh strength (Spec WP4 task 1/5). When the population model is
    # broadcast back down, each cohort is BLENDED toward it rather than
    # overwritten, so centre-specific learning is preserved:
    #   cohort <- (1 - refresh_blend) * cohort + refresh_blend * population
    # refresh_blend=1.0 reproduces the old full-overwrite behaviour (which
    # collapsed hier_cohort onto population); 0.0 disables top-down pull.
    refresh_blend: float = 0.3
    # Spec WP4 task 3 defines TWO upward mechanisms — patient->cohort and
    # cohort->population — so the weighting scheme is selectable per level. One
    # shared knob (the previous design) applies the same scheme to both and
    # therefore cannot compare them, which is what the task asks for.
    # ``weighting_scheme`` remains the default for both when the per-level knobs
    # are left unset.
    weighting_scheme: str = "volume"
    patient_weighting_scheme: str | None = None    # patient -> cohort
    cohort_weighting_scheme: str | None = None     # cohort  -> population
    # Half-life for time-based ``recency`` weighting, in days.
    recency_half_life_days: float = 180.0
    # Which patients each round draws on. False (default) takes the SAME first
    # ``patients_per_round`` streams every round, so any patient beyond that
    # prefix is built and never trained on, and upward propagation sees one
    # fixed group repeatedly. True walks a rotating window through the whole
    # stream list, so over R rounds up to ``R * patients_per_round`` distinct
    # patients feed the cohort at identical per-round cost.
    #
    # This changes the method, not just its measurement precision — leave it off
    # for runs meant to be compared against the v5 results.
    patient_rotation: bool = False
    # TD baseline distinction. Hierarchical (False): a patient adapter starts
    # from its centre's cohort state, inheriting centre-level knowledge — the
    # hierarchy's DOWNWARD flow. Local fine-tuning (True): the patient starts
    # from the population model instead, so no hierarchy is used in either
    # direction, which is what "per-centre and per-patient fine-tuning without
    # coordination" means. Without this the two baselines are the same method:
    # patients always initialise from the cohort state, so setting eta_cohort=0
    # on the local baseline would make it identical to top_down_only.
    patient_init_from_population: bool = False
    # Route steps 1-3 through Flower clients and step 4 through a Flower
    # strategy instead of the in-process loop. The arithmetic is identical
    # (tests/test_flower_equivalence.py); what changes is EXECUTION ORDER, since
    # a Flower client completes all of its own work inside fit() whereas the
    # in-process path runs every centre's cohort step before any patient step.
    # That reorders consumption of torch's global RNG, so cells agree within the
    # documented tolerance rather than bit-exactly.
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

        # Initialise per-cohort adapter states (start fresh per hospital)
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
        # Pristine identity states, kept so a patient can be initialised from
        # the population model rather than from its centre (see
        # ``patient_init_from_population``). The FFN adapter is identity at
        # initialisation, so this state IS the population model.
        self._initial_cohort_states = {
            hid: {k: v.clone() for k, v in st.items()}
            for hid, st in self.cohort_states.items()
        }
        # Population adapter is None until propagated up for the first time
        self.population_adapter_state: dict[str, torch.Tensor] | None = None
        self.round_idx = 0
        self.history: list[dict] = []
        # Patient updates applied since the last population sync — drives the
        # "after N patient updates" trigger (Spec WP4 task 5).
        self.patients_since_sync = 0
        # Set when a population sync happens; consumed at the START of the next
        # round (see run_round) so evaluation never sees a refreshed state for the
        # round that was just trained.
        self._refresh_pending = False
        # Flower-path state: per-round log fragments and pre-patient snapshots the
        # clients fill in, plus lazily built clients/strategy.
        self._flower_logs: dict = {}
        self._flower_pre_patient: dict = {}
        self._flower_refresh_pending = False
        self._flower_clients = None
        self._flower_strategy = None

        # Replay buffer for "replay" CL strategy: pre-populated from the
        # cohort training loaders (proxy for population/historic data).
        # One shared buffer across all hospitals so a cohort sees diverse
        # samples — keeps the adapter from over-specialising to its silo.
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
        # Blend (do NOT overwrite) the population adapter into each cohort so the
        # centre keeps the adaptation it just learned. Overwriting here is what
        # collapsed hier_cohort onto the population baseline (delta_cohort ~= 0).
        beta = self.cfg.refresh_blend
        if beta <= 0.0:
            return
        for hid in self.cohort_states:
            self._refresh_one_cohort(hid, self.population_adapter_state, beta)

    def _refresh_one_cohort(self, hid, population_state, beta=None) -> None:
        """Blend the population adapter into ONE centre's cohort state.

        Split out from ``_refresh_cohort_from_population`` because under Flower
        this is the client's job: the server broadcasts the population adapter and
        each centre blends it into its own state on receipt. Same arithmetic,
        performed where it would actually happen in a deployment.
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

        This is what makes ``recency`` mean recency: the weight of a patient's
        update decays with how old that patient's data is, rather than with where
        the patient happened to sit in the within-round list. Returns None when
        the dataset does not expose ``times``/``start_indices``, in which case
        ``_weights_for`` warns and falls back.
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

        Without rotation this is the leading ``patients_per_round`` streams,
        every round. With rotation the window advances by that many each round
        and wraps, so round r starts at ``(r-1) * patients_per_round``. When the
        window is at least as long as the stream list the two are equivalent —
        every patient is used either way.
        """
        n_per_round = self.cfg.patients_per_round
        n = len(patient_streams)
        if not self.cfg.patient_rotation or n_per_round >= n:
            return patient_streams[:n_per_round]
        start = ((self.round_idx - 1) * n_per_round) % n
        return [patient_streams[(start + k) % n] for k in range(n_per_round)]

    def _cohort_step(self, hid) -> dict:
        """Cohort-level training for ONE centre (round step 1).

        Extracted from ``run_round`` so the single-process path and the Flower
        client execute the SAME code rather than two copies. Re-implementing a
        per-centre step inside the client is exactly how the "two independent
        copies drifted apart" defect arose once already, so the client calls this.
        """
        loader = self.hospital_train_loaders[hid]
        tmpl = copy.deepcopy(self.population_clf)
        attach_ffn_adapter(tmpl.backbone, bottleneck=self.cfg.cohort_cfg.bottleneck,
                           attr="cohort_adapter")
        # Load current cohort state into the fresh template
        load_level_state_dict(tmpl, self.cohort_states[hid], attr="cohort_adapter")
        # Train one local round on top of current state
        for p in tmpl.backbone.cohort_adapter.parameters():
            p.requires_grad_(True)
        for p in tmpl.classifier.parameters():
            p.requires_grad_(self.cfg.cohort_cfg.train_head)
        if (self.cfg.cohort_cfg.train_head_bias_only
                and not self.cfg.cohort_cfg.train_head):
            # Intercepts only — see CohortConfig.train_head_bias_only.
            from ..adapters.ffn import head_bias_parameters
            for _n, p in head_bias_parameters(tmpl):
                p.requires_grad_(True)
        # When ``cohort_cfg.cl_strategy == "distill"`` the frozen population
        # classifier is the teacher, so the cohort adapter does not drift too far
        # from population knowledge (Spec WP4 task 4).
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
        """Patient training + patient->cohort propagation for ONE centre (steps 2-3).

        ``pre_state`` is this centre's cohort state as it stood after the cohort
        step and before any patient feedback — the reference the delta norm is
        measured against (Spec WP5 axis 4).
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
            # Real per-patient task loss; ``uncertainty`` weighting takes 1/loss.
            losses.append(p_loss)
            # Data recency for this patient (Spec WP4 task 3 "update recency").
            times.append(self._latest_time_in(p_train_loader))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Newest data this centre contributed, for cohort->population recency.
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
        # L2 norm of the cohort delta from patient feedback.
        delta_norm = 0.0
        for k, v_new in self.cohort_states[hid].items():
            if k in pre_state:
                delta_norm += float((v_new - pre_state[k]).pow(2).sum())
        log[f"cohort_{hid}_patient_delta_norm"] = delta_norm ** 0.5
        return log

    def run_round(self) -> dict:
        """One super-round: cohort -> patients -> upward -> (maybe) population update."""
        self.round_idx += 1
        round_log = {"round": self.round_idx}

        # Deferred top-down refresh from the previous round's population sync.
        # Done HERE so it seeds this round's training rather than altering the
        # previous round's already-reported state.
        if getattr(self, "_refresh_pending", False):
            self._refresh_cohort_from_population()
            self._refresh_pending = False
            round_log["refreshed_from_population"] = True

        # --- Steps 1-3: per-centre work, in-process or through Flower ---
        if self.cfg.use_flower:
            cohort_pre_patient = self._flower_centre_rounds(round_log)
            round_log["cohort_state_pre_patient"] = cohort_pre_patient
            return self._finish_round(round_log)

        # --- Step 1: cohort-level training (per centre) ---
        for hid in self.hospital_train_loaders:
            round_log.update(self._cohort_step(hid))
        # Spec WP5 axis 4: record cohort-state SNAPSHOT before patient aggregation
        # so the harness can attribute later cohort improvement to patient
        # feedback (vs the cohort-only training step that came before).
        cohort_pre_patient = {
            hid: {k: v.clone() for k, v in cs.items()}
            for hid, cs in self.cohort_states.items()
        }
        round_log["cohort_state_pre_patient"] = cohort_pre_patient

        # --- Step 2 + 3: patient training + upward propagation per cohort ---
        for hid in self.hospital_patient_streams:
            round_log.update(self._patient_step(hid, cohort_pre_patient[hid]))

        return self._finish_round(round_log)

    # ------------------------------------------------------------------ Flower
    def _flower_centre_rounds(self, round_log: dict) -> dict:
        """Steps 1-3 for every centre, executed as Flower clients.

        Each client completes its OWN cohort step and patient loop inside
        ``fit()`` — which is what a real federated client does, and why execution
        order differs from the in-process path. The clients call the very same
        ``_cohort_step`` / ``_patient_step`` used above rather than reimplementing
        them, so the two paths cannot drift apart.
        """
        from ..federated.flower_app import (
            make_hierarchical_client_class, state_to_ndarrays,
        )
        if getattr(self, "_flower_clients", None) is None:
            Client = make_hierarchical_client_class()
            self._flower_clients = {
                hid: Client(self, hid) for hid in self.hospital_train_loaders
            }
        # The server broadcasts the population adapter; each client blends it in
        # on receipt when a refresh is pending (the hierarchy's downward flow).
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
        """Step 4 through Flower's strategy rather than a direct call.

        ``HierarchicalEMAStrategy`` subclasses ``flwr.server.strategy.FedAvg`` and
        replaces its average with the cohort->population EMA blend, because stock
        FedAvg cannot express it (with eta=0.3 onto a zero cohort it returns 0.5
        where the blend returns 0.3). Verified in test_flower_equivalence.py.
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
        # Say it out loud. Three separate "green" bring-up runs tonight executed
        # less than they appeared to (a dropped flag, a stale file, and a trigger
        # that never fired), each reporting CELL OK. The cohort->population step
        # is the upward half of this project's central claim, so a run must state
        # in its log that it happened and by which route — not leave it to be
        # inferred from the absence of a crash.
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
        # --- Step 4: cohort -> population (slow) ---
        # --- Step 4 trigger (Spec WP4 task 5): round frequency, OR "after N
        # patient updates", OR divergence. The first condition met wins.
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
            # L2 distance between each cohort state and the last broadcast
            # population state. If any cohort exceeds the threshold the global
            # has drifted enough — trigger an immediate population update.
            #
            # Before the first population update the population adapter is the
            # ZERO state (an FFN adapter starts as an identity residual), not
            # "missing". This used to also require
            # ``population_adapter_state is not None``, which meant divergence
            # could never fire until a FREQUENCY update had already happened —
            # so at the default ``population_update_freq == max_rounds`` the
            # divergence arm never fired at all and a divergence-configured run
            # was silently identical to frequency-only.
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
            # Per-cohort task loss from this round's cohort-training step, in the
            # same hospital order as ``cohort_states``/``cohort_counts``.
            # ``weighting_scheme="uncertainty"`` takes 1/loss, so a centre whose
            # adapter fits its own data well contributes more to the population.
            #
            # Omitting this is what made the whole `uncertainty` arm crash:
            # ``_weights_for`` raises "uncertainty weighting requires losses"
            # when it gets None, and the 2026-08-08 fix only wired losses into
            # the patient->cohort call, not this cohort->population one.
            cohort_losses = [
                round_log.get(f"cohort_{h}_loss") if
                round_log.get(f"cohort_{h}_loss") is not None else 1.0
                for h in self.cohort_states
            ]
            # Recency at this level is the newest data any of the cohort's own
            # patients contributed, so a centre that has not seen recent data
            # decays. Falls back to None (and a warning) if unavailable.
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
            # Reset the patient-update budget only when a sync actually happens.
            self.patients_since_sync = 0

            # --- Step 5: top-down refresh — DEFERRED to the next round ---
            # The refresh re-seeds cohorts FROM the updated population, i.e. it is
            # the starting point for round r+1. Applying it here and returning
            # meant the caller evaluated round r on state that had already been
            # pulled toward the population — attenuating the very thing being
            # measured, and only for `hierarchical` (the other methods push the
            # sync out of reach and never refresh).
            #
            # The magnitude is not small. At the first sync the population adapter
            # does not exist yet, so propagate_cohorts_to_population returns
            # eta_population * mean(cohorts) = 0.1 * mean. The refresh then gives
            #   cohort <- 0.7*cohort + 0.3*(0.1*mean) ~= 0.73 * cohort
            # and the FFN adapter is a RESIDUAL that is identity at init, so
            # scaling it by 0.73 moves the model ~27 % back toward the UNADAPTED
            # population model, immediately before the reported evaluation.
            # `hierarchical` selecting round 1 in every completed cell, while
            # fedavg selected round 5, is what that looks like from the outside.
            if self.cfg.refresh_cohort_from_population:
                if self.cfg.use_flower:
                    # Applied by each client on receipt of the broadcast, next round.
                    self._flower_refresh_pending = True
                else:
                    self._refresh_pending = True

        self.history.append(round_log)
        return round_log
