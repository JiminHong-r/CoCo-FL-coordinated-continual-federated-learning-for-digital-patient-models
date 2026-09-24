"""WP5 baselines (TD section 2.5), with model selection and full metrics.

The four comparison points for the hierarchical method:

1. ``eval_population_only``: the frozen population model, no training. 2.
``run_local_finetune``: per-centre and per-patient fine-tuning with no coordination.
Patients start from the population model. 3. ``run_standard_fedavg``: plain FedAvg
over one shared cohort adapter. 4. ``run_top_down_only``: the hierarchy without
upward propagation. Patients start from their centre's cohort model, so comparing
with (2) measures the value of the downward flow.

Each training method returns a :class:`MethodResult` with per-hospital AUROC, AUPRC
and ECE, rounds used, wall-clock time, the validation trajectory and method-specific
extras.
"""
from __future__ import annotations

import copy
import time
from typing import Callable, NamedTuple

import torch
from torch.utils.data import DataLoader

from ..adapters import (
    attach_ffn_adapter, level_state_dict, load_level_state_dict,
)
from ..hierarchical.coordinator import CoordinatorConfig, HierarchicalCoordinator
from ..models.classifier import TFNDownstreamClassifier
from ..personalisation.evaluate import eval_full
from .early_stopping import EarlyStoppingMonitor
from .metrics import macro_average
from .result import MethodResult


def _snapshot(states: dict[int, dict]) -> dict[int, dict]:
    """Deep copy of per-hospital states, for best-round model selection."""
    return {hid: {k: v.clone() for k, v in st.items()} for hid, st in states.items()}


def state_delta_norm(states: dict[int, dict], reference: dict[int, dict]) -> float:
    """L2 norm of ``states - reference`` over every hospital's adapter tensors.

    Records how far model selection moved from round 0, which separates "adaptation
    changed nothing" from "the weights changed but the eval set could not see it".
    """
    total = 0.0
    for hid, st in states.items():
        ref = reference.get(hid, {})
        for k, v in st.items():
            r = ref.get(k)
            d = (v - r) if r is not None else v
            total += float(d.pow(2).sum())
    return total ** 0.5


class Selection(NamedTuple):
    """What model selection actually installed."""
    states: dict[int, dict]
    val_auroc: float | None
    round: int | None                       # None when centres chose different rounds
    per_hospital_round: dict[int, int] | None


def resolve_selection(*, per_hospital_selection: bool,
                      global_states: dict[int, dict], global_val: float | None,
                      global_round: int | None,
                      per_hosp_state: dict[int, dict],
                      per_hosp_best: dict[int, float | None],
                      per_hosp_round: dict[int, int]) -> Selection:
    """Make the reported selection numbers match the states actually installed.

    Under per-centre selection each centre keeps its own best round. The reported
    validation AUROC is then the mean of those per-centre bests (not the best global
    round), and ``round`` is None because no single round applies; the per-centre
    rounds are returned separately. Without this, a row could say
    ``selected_round=0`` while the weights had clearly moved.

    Per-centre selection maximises over rounds separately for each centre, so its
    validation number is optimistic compared with one global choice; expect a wider
    gap between validation and test.
    """
    if not per_hospital_selection:
        return Selection(global_states, global_val, global_round, None)
    val = macro_average(list(per_hosp_best.values()))
    print(f"    per-hospital selection: rounds {per_hosp_round}  "
          f"val macro {'—' if val is None else f'{val:.4f}'} "
          f"(global-choice reference "
          f"{'—' if global_val is None else f'{global_val:.4f}'} @ round {global_round})")
    return Selection(per_hosp_state, val, None, dict(per_hosp_round))


def _eval_full_per_hospital(template: TFNDownstreamClassifier,
                              cohort_states: dict[int, dict],
                              eval_loaders: dict[int, DataLoader],
                              bottleneck: int,
                              label_fn: Callable,
                              device: torch.device) -> dict[int, dict[str, float | None]]:
    """Return ``{hid: {"auc", "auprc", "ece", "n", "n_pos"}}`` per hospital.

    The sizes come with the metrics so every method can report the eval-set size
    behind its AUROC, including runs where only ``hierarchical`` runs.
    """
    out: dict[int, dict[str, float | None]] = {}
    for hid, state in cohort_states.items():
        if hid not in eval_loaders:
            continue
        tmpl = copy.deepcopy(template)
        attach_ffn_adapter(tmpl.backbone, bottleneck=bottleneck, attr="cohort_adapter")
        load_level_state_dict(tmpl, state, attr="cohort_adapter")
        m = eval_full(tmpl, eval_loaders[hid], label_fn, device)
        out[hid] = {"auc": m.auroc, "auprc": m.auprc, "ece": m.ece,
                    "n": m.n, "n_pos": m.n_pos,
                    "preds": m.preds, "labels": m.labels}
        del tmpl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def evaluate_patient_layer(
    population_clf: TFNDownstreamClassifier,
    cohort_states: dict[int, dict],
    patient_streams: dict[int, list],
    label_fn: Callable,
    device: torch.device,
    coord_cfg: CoordinatorConfig,
    patient_eval_n: int,
    patient_min_pos: int = 10,
    init_states: dict[int, dict] | None = None,
) -> dict:
    """Train, accept or reject, and score the patient level for one method.

    Shared by ``hierarchical`` and the local fine-tuning baseline so both are judged
    by the same rule.

    ``init_states`` is where each patient adapter starts. The cohort states (the
    default) give hierarchical behaviour: a patient inherits its centre's knowledge.
    The initial identity states make patients start from the population model, as
    the TD's "local fine-tuning without coordination" baseline requires. The gap
    between the two is the value of the downward flow.
    """
    from ..hierarchical.patient import train_patient_adapter
    from ..personalisation.evaluate import auc_from_preds, predict

    base_states = init_states if init_states is not None else cohort_states
    # ``cohort_aucs`` scores the un-adapted starting model on the same pooled
    # patient windows, so Delta_patient = Perf_patient - Perf_cohort is a paired
    # difference on identical data.
    out = {"aucs": {}, "cohort_aucs": {}, "accepted": {}, "considered": {},
           "unevaluable": {}, "pooled_pos": {}, "records": []}
    for hid, streams in patient_streams.items():
        if not streams or hid not in cohort_states:
            out["aucs"][hid] = None
            out["cohort_aucs"][hid] = None
            for k in ("accepted", "considered", "unevaluable", "pooled_pos"):
                out[k][hid] = 0
            continue
        start_state = base_states.get(hid, cohort_states[hid])
        pooled_preds: list[float] = []
        pooled_labels: list[int] = []
        base_pooled_preds: list[float] = []
        base_pooled_labels: list[int] = []
        n_kept = n_unevaluable = 0
        for pid, pt_train, pt_val, pt_eval in streams[:patient_eval_n]:
            pers_clf, *_ = train_patient_adapter(
                population_clf, start_state, pt_train, label_fn,
                coord_cfg.patient_cfg, device,
                cohort_bottleneck=coord_cfg.cohort_cfg.bottleneck,
            )
            # The reference this adapter must beat: the same starting state without
            # patient adaptation.
            base_clf = copy.deepcopy(population_clf)
            attach_ffn_adapter(base_clf.backbone,
                               bottleneck=coord_cfg.cohort_cfg.bottleneck,
                               attr="cohort_adapter")
            load_level_state_dict(base_clf, start_state, attr="cohort_adapter")
            # Accept or reject on the patient's validation slice, never on the eval
            # slice.
            pv, yv = predict(pers_clf, pt_val, label_fn, device)
            bv, ybv = predict(base_clf, pt_val, label_fn, device)
            a_pers = auc_from_preds(pv, yv).auc
            a_base = auc_from_preds(bv, ybv).auc
            evaluable = a_pers is not None
            use_patient = evaluable and (a_base is None or a_pers >= a_base)
            n_unevaluable += int(not evaluable)
            n_kept += int(use_patient)
            chosen = pers_clf if use_patient else base_clf
            p, y = predict(chosen, pt_eval, label_fn, device)
            pooled_preds.extend(p); pooled_labels.extend(y)
            # Paired reference: the un-adapted model on exactly the same windows.
            bp, by = predict(base_clf, pt_eval, label_fn, device)
            base_pooled_preds.extend(bp); base_pooled_labels.extend(by)
            # One row per patient, for axis 2 ("when and for which patient subgroups
            # personalisation is most beneficial"). This cannot be rebuilt after a
            # run.
            out["records"].append({
                "hospital": hid,
                "patient_id": pid,
                "n_fit": len(pt_train.dataset),
                "n_val": len(pt_val.dataset),
                "n_eval": len(pt_eval.dataset),
                "n_eval_pos": int(sum(y)),
                "val_auroc_patient": a_pers,
                "val_auroc_cohort": a_base,
                "evaluable": bool(evaluable),
                "accepted": bool(use_patient),
            })
            del pers_clf, base_clf
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        n_considered = len(streams[:patient_eval_n])
        print(f"    patient selection h{hid}: kept {n_kept}/{n_considered} "
              f"({n_unevaluable} unevaluable — single-class val slice)")
        n_pos = sum(pooled_labels)
        n_neg = len(pooled_labels) - n_pos
        out["accepted"][hid] = n_kept
        out["considered"][hid] = n_considered
        out["unevaluable"][hid] = n_unevaluable
        out["pooled_pos"][hid] = n_pos
        if n_pos < patient_min_pos or n_neg < patient_min_pos:
            print(f"      patient AUROC withheld for h{hid}: pooled set has "
                  f"{n_pos} pos / {n_neg} neg, floor is {patient_min_pos}")
            out["aucs"][hid] = None
            out["cohort_aucs"][hid] = None
        else:
            out["aucs"][hid] = auc_from_preds(pooled_preds, pooled_labels).auc
            out["cohort_aucs"][hid] = auc_from_preds(
                base_pooled_preds, base_pooled_labels).auc
            print(f"      Δ_patient (paired, same windows): patient="
                  f"{out['aucs'][hid]} cohort={out['cohort_aucs'][hid]}")
    return out


def eval_population_only(
    population_clf: TFNDownstreamClassifier,
    eval_loaders: dict[int, DataLoader],
    label_fn: Callable,
    device: torch.device,
) -> MethodResult:
    """Score the frozen population model: no training, no model selection."""
    res = MethodResult()
    t0 = time.perf_counter()
    population_clf.eval()
    for hid, loader in eval_loaders.items():
        m = eval_full(population_clf, loader, label_fn, device)
        res.aucs[hid] = m.auroc
        res.auprcs[hid] = m.auprc
        res.eces[hid] = m.ece
        # Every method uses the same loaders, so these sizes describe the whole
        # cell.
        res.ns[hid] = m.n
        res.n_pos[hid] = m.n_pos
        res.extra.setdefault("test_preds", {})[hid] = m.preds
        res.extra.setdefault("test_labels", {})[hid] = m.labels
    res.wall_clock_sec = time.perf_counter() - t0
    return res


def _round_loop_with_early_stopping(
    coord: HierarchicalCoordinator,
    val_loaders: dict[int, DataLoader],
    test_loaders: dict[int, DataLoader],
    bottleneck: int,
    label_fn: Callable,
    device: torch.device,
    max_rounds: int,
    patience: int,
    min_rounds: int,
    per_hospital_selection: bool = False,
    probe_loaders: dict[int, DataLoader] | None = None,
) -> MethodResult:
    """Round loop shared by local_finetune and top_down_only.

    By default one round is chosen for all centres, by the macro average of their
    validation AUROCs, so the round that suits the biggest centre is imposed on the
    rest. ``per_hospital_selection`` lets each centre keep its own best round
    instead; under the class-aware split the centres differ a lot, so their best
    rounds need not match. Round 0 stays a candidate for every centre. Off by
    default: it changes the method.
    """
    monitor = EarlyStoppingMonitor(patience=patience, min_rounds=min_rounds)
    t0 = time.perf_counter()
    # Round 0 is the un-adapted state (the adapter starts as the identity), i.e. the
    # population model. Keeping it as a candidate means a method is never reported
    # below where it started.
    init_states = _snapshot(coord.cohort_states)
    best_states, best_val, best_round = _snapshot(coord.cohort_states), None, 0
    val0_per_hosp = _eval_full_per_hospital(
        coord.population_clf, coord.cohort_states, val_loaders,
        bottleneck, label_fn, device)
    val0 = macro_average([v["auc"] for v in val0_per_hosp.values()])
    best_val = val0
    # Per-centre bookkeeping, used only when per_hospital_selection is on.
    per_hosp_best = {hid: v["auc"] for hid, v in val0_per_hosp.items()}
    per_hosp_round = {hid: 0 for hid in val0_per_hosp}
    per_hosp_state = {hid: {k: v.clone() for k, v in st.items()}
                      for hid, st in coord.cohort_states.items()}
    for r in range(1, max_rounds + 1):
        coord.run_round()
        val_per_hosp = _eval_full_per_hospital(
            coord.population_clf, coord.cohort_states, val_loaders,
            bottleneck, label_fn, device,
        )
        val_macro = macro_average([v["auc"] for v in val_per_hosp.values()])
        if val_macro is not None and (best_val is None or val_macro > best_val):
            best_val, best_states, best_round = val_macro, _snapshot(coord.cohort_states), r
        for hid, v in val_per_hosp.items():
            a = v["auc"]
            if a is None:
                continue
            if per_hosp_best.get(hid) is None or a > per_hosp_best[hid]:
                per_hosp_best[hid] = a
                per_hosp_round[hid] = r
                per_hosp_state[hid] = {k: t.clone()
                                       for k, t in coord.cohort_states[hid].items()}
        if monitor.update(r, val_macro):
            break
    wall_clock = time.perf_counter() - t0

    _sel = resolve_selection(
        per_hospital_selection=per_hospital_selection,
        global_states=best_states, global_val=best_val, global_round=best_round,
        per_hosp_state=per_hosp_state, per_hosp_best=per_hosp_best,
        per_hosp_round=per_hosp_round,
    )
    best_states, best_val, best_round = _sel.states, _sel.val_auroc, _sel.round

    coord.cohort_states = best_states
    delta_norm = state_delta_norm(best_states, init_states)
    print(f"    model selection: best val macro AUROC {best_val:.4f} @ round "
          f"{'per-centre' if best_round is None else best_round}"
          f" (|Δ| from round 0 = {delta_norm:.3e})"
          if best_val is not None else "    model selection: val undefined")
    test_per_hosp = _eval_full_per_hospital(
        coord.population_clf, coord.cohort_states, test_loaders,
        bottleneck, label_fn, device,
    )
    probe_auroc = None
    if probe_loaders:
        probe_auroc = macro_average([
            v["auc"] for v in _eval_full_per_hospital(
                coord.population_clf, coord.cohort_states, probe_loaders,
                bottleneck, label_fn, device).values()])
    res = MethodResult(
        aucs={hid: v["auc"] for hid, v in test_per_hosp.items()},
        auprcs={hid: v["auprc"] for hid, v in test_per_hosp.items()},
        eces={hid: v["ece"] for hid, v in test_per_hosp.items()},
        ns={hid: v["n"] for hid, v in test_per_hosp.items()},
        n_pos={hid: v["n_pos"] for hid, v in test_per_hosp.items()},
        n_rounds=monitor.last_round,
        selected_round=best_round,
        selected_delta_norm=delta_norm,
        val_auroc=best_val,
        val_auroc_round0=val0,
        probe_auroc=probe_auroc,
        wall_clock_sec=wall_clock,
        val_history=monitor.history,
        extra={"selected_round_per_hospital": dict(per_hosp_round),
               "test_preds": {h: v["preds"] for h, v in test_per_hosp.items()},
               "test_labels": {h: v["labels"] for h, v in test_per_hosp.items()}},
    )
    return res


def run_local_finetune(
    population_clf: TFNDownstreamClassifier,
    hospital_train_loaders: dict[int, DataLoader],
    hospital_val_loaders: dict[int, DataLoader],
    hospital_test_loaders: dict[int, DataLoader],
    label_fn: Callable,
    device: torch.device,
    coord_cfg: CoordinatorConfig,
    max_rounds: int = 15,
    patience: int = 3,
    min_rounds: int = 2,
    per_hospital_selection: bool = False,
    patient_streams: dict[int, list] | None = None,
    patient_eval_n: int | None = None,
    patient_min_pos: int = 10,
    probe_loaders: dict[int, DataLoader] | None = None,
) -> MethodResult:
    """Per-centre and per-patient fine-tuning, without coordination (the TD baseline).

    "Without coordination" is enforced three ways: no cohort-to-population sync (the
    sync frequency is set out of reach), no patient-to-cohort aggregation
    (``eta_cohort=0``), and patients start from the population model instead of
    their centre. The last one is what makes this different from ``top_down_only``.
    """
    isolated_cfg = copy.copy(coord_cfg)
    isolated_cfg.population_update_freq = max(max_rounds + 10, 100)
    isolated_cfg.eta_cohort = 0.0
    isolated_cfg.eta_population = 0.0
    isolated_cfg.patient_init_from_population = True
    coord = HierarchicalCoordinator(
        population_clf, hospital_train_loaders,
        hospital_patient_streams=(
            patient_streams if patient_streams is not None
            else {hid: [] for hid in hospital_train_loaders}),
        label_fn=label_fn, device=device, cfg=isolated_cfg,
    )
    res = _round_loop_with_early_stopping(
        coord, hospital_val_loaders, hospital_test_loaders,
        isolated_cfg.cohort_cfg.bottleneck, label_fn, device,
        max_rounds, patience, min_rounds,
        per_hospital_selection=per_hospital_selection,
        probe_loaders=probe_loaders,
    )
    if patient_streams:
        pl = evaluate_patient_layer(
            population_clf, coord.cohort_states, patient_streams, label_fn,
            device, isolated_cfg, patient_eval_n or 1, patient_min_pos,
            init_states=coord._initial_cohort_states,
        )
        res.extra.update({
            "patient_aucs": pl["aucs"],
            "patient_accepted": pl["accepted"],
            "patient_considered": pl["considered"],
            "patient_unevaluable": pl["unevaluable"],
            "patient_pooled_pos": pl["pooled_pos"],
        })
    return res


def run_top_down_only(
    population_clf: TFNDownstreamClassifier,
    hospital_train_loaders: dict[int, DataLoader],
    hospital_val_loaders: dict[int, DataLoader],
    hospital_test_loaders: dict[int, DataLoader],
    hospital_patient_streams: dict[int, list[tuple[DataLoader, DataLoader]]],
    label_fn: Callable,
    device: torch.device,
    coord_cfg: CoordinatorConfig,
    max_rounds: int = 15,
    patience: int = 3,
    min_rounds: int = 2,
    per_hospital_selection: bool = False,
    probe_loaders: dict[int, DataLoader] | None = None,
) -> MethodResult:
    """Top-down only: the hierarchy without upward propagation."""
    no_upward_cfg = copy.copy(coord_cfg)
    no_upward_cfg.eta_cohort = 0.0
    no_upward_cfg.eta_population = 0.0
    no_upward_cfg.population_update_freq = max(max_rounds + 10, 100)
    coord = HierarchicalCoordinator(
        population_clf, hospital_train_loaders, hospital_patient_streams,
        label_fn=label_fn, device=device, cfg=no_upward_cfg,
    )
    return _round_loop_with_early_stopping(
        coord, hospital_val_loaders, hospital_test_loaders,
        no_upward_cfg.cohort_cfg.bottleneck, label_fn, device,
        max_rounds, patience, min_rounds,
        per_hospital_selection=per_hospital_selection,
        probe_loaders=probe_loaders,
    )


def run_standard_fedavg(
    population_clf: TFNDownstreamClassifier,
    hospital_train_loaders: dict[int, DataLoader],
    hospital_val_loaders: dict[int, DataLoader],
    hospital_test_loaders: dict[int, DataLoader],
    label_fn: Callable,
    device: torch.device,
    coord_cfg: CoordinatorConfig,
    max_rounds: int = 15,
    patience: int = 3,
    min_rounds: int = 2,
    use_cl: bool = True,
    probe_loaders: dict[int, DataLoader] | None = None,
) -> MethodResult:
    """Plain FedAvg over one shared cohort adapter.

    Returns a MethodResult with the final server state in ``extra["server_state"]``.

    ``use_cl`` decides whether FedAvg gets the same continual-learning regulariser
    (distillation towards the population model) as the other methods. With it, the
    comparison isolates the aggregation topology; without it, this is vanilla
    FedAvg. The function defaults to True, but the harness passes False, because the
    TD defines the baseline as FedAvg "without continual learning or hierarchy".
    """
    from ..hierarchical.cohort import _train_one_cohort_step
    from ..hierarchical.upward_prop import weighted_state_average

    template = copy.deepcopy(population_clf)
    attach_ffn_adapter(template.backbone,
                       bottleneck=coord_cfg.cohort_cfg.bottleneck,
                       attr="cohort_adapter")
    # Include the classifier head in the state, like the other methods do. FedAvg
    # trains the head below, so saving only the adapter would score it against the
    # wrong head. Every method must treat the head the same way.
    server_state = level_state_dict(template, attr="cohort_adapter",
                                    include_head=coord_cfg.cohort_cfg.train_head)

    # Same regulariser as the coordinator-based methods, when use_cl is on.
    teacher = population_clf if (use_cl and coord_cfg.cohort_cfg.cl_strategy == "distill") else None
    replay_buffer = None
    if use_cl and coord_cfg.cohort_cfg.cl_strategy == "replay":
        from ..personalisation.continual import ReplayBufferTFN
        replay_buffer = ReplayBufferTFN(capacity=coord_cfg.cohort_cfg.cl_replay_capacity)
        for _ld in hospital_train_loaders.values():
            replay_buffer.fill_from(_ld, label_fn)
            if len(replay_buffer.items) >= replay_buffer.capacity:
                break

    monitor = EarlyStoppingMonitor(patience=patience, min_rounds=min_rounds)
    t0 = time.perf_counter()
    # Round 0 = identity adapter = the population model; a valid candidate.
    init_state = {k: v.clone() for k, v in server_state.items()}
    val0_fed = None
    best_state = {k: v.clone() for k, v in server_state.items()}
    best_val = macro_average([v["auc"] for v in _eval_full_per_hospital(
        template, {hid: server_state for hid in hospital_train_loaders},
        hospital_val_loaders, coord_cfg.cohort_cfg.bottleneck, label_fn, device,
    ).values()])
    val0_fed = best_val
    best_round = 0
    def _local_step(hid, state):
        """One centre's local training for one round: state -> (state, n_seen).

        Used by both runtimes: the in-process loop calls it directly and the Flower
        client calls it through ``fit``. One definition keeps the two paths the
        same.
        """
        loader = hospital_train_loaders[hid]
        clf = copy.deepcopy(template)
        load_level_state_dict(clf, state, attr="cohort_adapter")
        for p in clf.backbone.cohort_adapter.parameters():
            p.requires_grad_(True)
        for p in clf.classifier.parameters():
            p.requires_grad_(coord_cfg.cohort_cfg.train_head)
        n_seen, _ = _train_one_cohort_step(
            clf, loader, label_fn, coord_cfg.cohort_cfg, device,
            teacher=teacher, replay_buffer=replay_buffer,
        )
        out = level_state_dict(
            clf, attr="cohort_adapter",
            include_head=coord_cfg.cohort_cfg.train_head)
        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out, n_seen

    # standard_fedavg is the only other method that crosses a centre boundary, so it
    # uses the same runtime as the hierarchy; the two are compared directly.
    use_flower = bool(getattr(coord_cfg, "use_flower", False))
    for r in range(1, max_rounds + 1):
        if use_flower:
            from ..federated.flower_app import flower_fedavg_round
            server_state, _counts = flower_fedavg_round(
                server_state,
                {hid: (lambda st, h=hid: _local_step(h, st))
                 for hid in hospital_train_loaders},
            )
        else:
            local_states: list[dict[str, torch.Tensor]] = []
            local_counts: list[float] = []
            for hid in hospital_train_loaders:
                st, n_seen = _local_step(hid, server_state)
                local_states.append(st)
                local_counts.append(float(max(n_seen, 1)))
            server_state = weighted_state_average(local_states, local_counts)
        broadcast_states = {hid: server_state for hid in hospital_train_loaders}
        val_per_hosp = _eval_full_per_hospital(
            template, broadcast_states, hospital_val_loaders,
            coord_cfg.cohort_cfg.bottleneck, label_fn, device,
        )
        val_macro = macro_average([v["auc"] for v in val_per_hosp.values()])
        if val_macro is not None and (best_val is None or val_macro > best_val):
            best_val = val_macro
            best_state = {k: v.clone() for k, v in server_state.items()}
            best_round = r
        if monitor.update(r, val_macro):
            break
    wall_clock = time.perf_counter() - t0

    server_state = best_state
    # One shared server state, so compare it once against round 0 instead of once
    # per hospital.
    delta_norm = state_delta_norm({0: server_state}, {0: init_state})
    print(f"    model selection: best val macro AUROC {best_val:.4f} @ round "
          f"{'per-centre' if best_round is None else best_round}"
          f" (|Δ| from round 0 = {delta_norm:.3e})"
          if best_val is not None else "    model selection: val undefined")
    broadcast_states = {hid: server_state for hid in hospital_train_loaders}
    test_per_hosp = _eval_full_per_hospital(
        template, broadcast_states, hospital_test_loaders,
        coord_cfg.cohort_cfg.bottleneck, label_fn, device,
    )
    probe_auroc_fed = None
    if probe_loaders:
        probe_auroc_fed = macro_average([
            v["auc"] for v in _eval_full_per_hospital(
                template, broadcast_states, probe_loaders,
                coord_cfg.cohort_cfg.bottleneck, label_fn, device).values()])
    res = MethodResult(
        aucs={hid: v["auc"] for hid, v in test_per_hosp.items()},
        auprcs={hid: v["auprc"] for hid, v in test_per_hosp.items()},
        eces={hid: v["ece"] for hid, v in test_per_hosp.items()},
        ns={hid: v["n"] for hid, v in test_per_hosp.items()},
        n_pos={hid: v["n_pos"] for hid, v in test_per_hosp.items()},
        n_rounds=monitor.last_round,
        selected_round=best_round,
        selected_delta_norm=delta_norm,
        val_auroc=best_val,
        val_auroc_round0=val0_fed,
        probe_auroc=probe_auroc_fed,
        wall_clock_sec=wall_clock,
        val_history=monitor.history,
        extra={"server_state": server_state,
               "test_preds": {h: v["preds"] for h, v in test_per_hosp.items()},
               "test_labels": {h: v["labels"] for h, v in test_per_hosp.items()}},
    )
    return res
