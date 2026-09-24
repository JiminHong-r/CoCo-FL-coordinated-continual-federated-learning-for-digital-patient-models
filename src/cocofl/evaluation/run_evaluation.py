"""WP5 evaluation harness: run every method on every (task, alpha) cell.

The five methods are population_only (the frozen model), local_finetune,
standard_fedavg, top_down_only and hierarchical (the full method). Each is scored
per hospital and as a macro average, and one row per hospital plus a MACRO row is
written to wp5_results.csv.

The defaults are small so a smoke run is quick. The reported grid sets its settings
through scripts/wp5_common.sh.

Usage::

    cocofl_evaluate \\
        model_path=$PROJECT/models/population/best_model.pt \\
        +train_dir=$PROJECT/data/tokenized/mimic_train \\
        input_dir=$PROJECT/data/tokenized/mimic_train \\
        output_dir=$PROJECT/wp5_artifacts \\
        +tasks=[hospital_mortality] +alphas=[0.1,1.0,10.0] \\
        +num_hospitals=3 +max_rounds=2 \\
        +cohort_local_steps=5 +n_patient_iters=5 \\
        +max_patients_per_hospital=3 +n_pop_eval=50
"""
from __future__ import annotations

import copy
import pickle
import random
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from ..constants import STATIC_DATA_FN, horizon_microseconds
from ..utils import dump_config
from ..data.static_features import build_static_arrays
from ..hierarchical.cohort import CohortConfig
from ..hierarchical.coordinator import CoordinatorConfig, HierarchicalCoordinator
from ..hierarchical.patient import PatientConfig
from ..hierarchical.run_hierarchical import (
    build_hospital_train_loader, build_patient_streams,
    build_population_classifier,
)
from ..adapters import attach_ffn_adapter, load_level_state_dict
from .batching import make_eval_loader
from ..personalisation.evaluate import eval_auc
from ..personalisation.run_ablation import POSITIVE_TOKENS, TASKS, make_collate, make_label_fn
from ..vocabulary import Vocabulary
from ..federated.hospital_split import dirichlet_split_patients
from .baselines import (
    eval_population_only, run_local_finetune, run_standard_fedavg,
    run_top_down_only,
)
from .result import MethodResult
from .metrics import (
    AucPerLevel, auroc_ci95_halfwidth, delta_cohort, delta_patient, fwt,
    macro_average, naive_full_model_parameters, parameters_updated, variance,
)


def _cap(idxs, n):
    """First ``n`` indices, or all of them when ``n`` is 0 or None.

    The indices are already in shuffled patient order, so a prefix is a fair
    subsample. Every method gets the same prefix, so comparisons stay paired.
    """
    return list(idxs) if not n else list(idxs)[:n]


def _build_pop_eval_loader(hospital_eval_loaders, test_ds, collate_fn, n_pop_eval,
                             batch_size=1):
    """Pooled reference set for the frozen population model.

    It is drawn from the hospitals' eval partitions, the same kind of data the
    cohort models are scored on, so forward transfer (cohort AUROC minus this
    baseline) compares like with like.
    """
    pop_eval_indices = sorted({
        idx for ld in hospital_eval_loaders.values()
        for idx in ld.dataset.indices
    })[:n_pop_eval]
    return make_eval_loader(test_ds, pop_eval_indices, collate_fn, batch_size=batch_size)


def _eval_sizes(*results) -> tuple[dict, dict]:
    """First non-empty ``(ns, n_pos)`` among ``results``.

    All methods score the same loaders, so the eval-set size belongs to the cell,
    not to one method. Falling back lets a run with only ``hierarchical`` (the WP4
    ablations) still report sizes and confidence intervals.
    """
    for r in results:
        if r.ns:
            return r.ns, r.n_pos
    return {}, {}


def _reportable_hospitals(hids, ns: dict, n_pos: dict,
                          min_pos: int, min_neg: int) -> list:
    """Hospitals whose eval set can support an AUROC.

    A hospital qualifies with at least ``min_pos`` positives and ``min_neg``
    negatives. Hospitals with unknown sizes are kept rather than silently dropped.
    """
    keep = []
    for hid in hids:
        n, npos = ns.get(hid), n_pos.get(hid)
        if n is None or npos is None:
            keep.append(hid)
            continue
        if npos >= min_pos and (n - npos) >= min_neg:
            keep.append(hid)
    return keep


def _macro_over(values_by_hid: dict, hids) -> float | None:
    """Macro-average restricted to ``hids`` (the MACRO-eligible hospitals)."""
    return macro_average([values_by_hid.get(h) for h in hids])


def _sum_or_none(counts_by_hid: dict, hids) -> int | None:
    """Sum of counts over ``hids``; ``None`` only when nothing was recorded.

    Zero must stay zero: "0 of 23 adapters accepted" is a result, while a blank
    would read as "the patient layer never ran".
    """
    present = [counts_by_hid.get(h) for h in hids if counts_by_hid.get(h) is not None]
    return sum(present) if present else None


def _fmt_per_hosp_round(res) -> str | None:
    """``"0:3 1:5 2:0"`` when centres selected independently, else None."""
    d = (res.extra or {}).get("selected_round_per_hospital")
    return " ".join(f"{h}:{r}" for h, r in sorted(d.items())) if d else None


def run_hierarchical_bidir(
    population_clf, hospital_train_loaders, hospital_patient_streams,
    label_fn, device, coord_cfg, eval_loaders,
    val_loaders=None, max_rounds=15, patience=3, min_rounds=2,
    traj_loaders=None, patient_eval_n=None, patient_min_pos=10,
    per_hospital_selection=False,
    pop_eval_loader=None, pop_baseline_auc=None, probe_loaders=None,
):
    """Run the full hierarchical method and return a MethodResult.

    ``eval_loaders`` is scored once, at the end, for the reported test metrics. The
    per-round trajectory (BWT, propagation gain) is scored on ``traj_loaders``
    instead, which can be a smaller subset: it costs two extra passes every round,
    and it only feeds round-to-round differences on a fixed sample.
    """
    from .early_stopping import EarlyStoppingMonitor
    from .metrics import macro_average
    from .result import MethodResult
    import time

    coord = HierarchicalCoordinator(
        population_clf, hospital_train_loaders, hospital_patient_streams,
        label_fn=label_fn, device=device, cfg=coord_cfg,
    )
    history: list[dict] = []
    per_round_cohort_aucs: list[dict[int, float | None]] = []
    monitor = EarlyStoppingMonitor(patience=patience, min_rounds=min_rounds)
    val_lds = val_loaders if val_loaders is not None else eval_loaders
    traj_lds = traj_loaders if traj_loaders is not None else eval_loaders
    t0 = time.perf_counter()

    propagation_gains: list[dict[int, float | None]] = []
    _init_states = {h: {k: v.clone() for k, v in st.items()}
                    for h, st in coord.cohort_states.items()}
    _best_states = {h: {k: v.clone() for k, v in st.items()}
                    for h, st in coord.cohort_states.items()}
    _best_round = 0
    _val0: dict[int, float | None] = {}
    for hid, state in coord.cohort_states.items():
        if hid not in val_lds:
            continue
        tmpl = copy.deepcopy(population_clf)
        attach_ffn_adapter(tmpl.backbone,
                           bottleneck=coord_cfg.cohort_cfg.bottleneck,
                           attr="cohort_adapter")
        load_level_state_dict(tmpl, state, attr="cohort_adapter")
        _val0[hid] = eval_auc(tmpl, val_lds[hid], label_fn, device).auc
        del tmpl
    _best_val = macro_average(_val0.values())
    # Score the population model after every round. Axis 3 (population stability)
    # and axis 4 (population gain from cohort aggregation) both need it.
    population_round_aucs: list[float | None] = []
    probe_round_aucs: list[float | None] = []
    _per_hosp_best = dict(_val0)
    _per_hosp_round = {h: 0 for h in _val0}
    _per_hosp_state = {h: {k: v.clone() for k, v in st.items()}
                       for h, st in coord.cohort_states.items()}
    for r in range(1, max_rounds + 1):
        log = coord.run_round()
        # Score each cohort after cohort training but before patient feedback, so
        # the change caused by patient feedback can be measured (axis 4).
        cohort_pre = log.pop("cohort_state_pre_patient", None)
        pre_patient_aucs: dict[int, float | None] = {}
        if cohort_pre is not None:
            for hid, pre_state in cohort_pre.items():
                if hid not in traj_lds:
                    pre_patient_aucs[hid] = None
                    continue
                tmpl = copy.deepcopy(population_clf)
                attach_ffn_adapter(tmpl.backbone,
                                   bottleneck=coord_cfg.cohort_cfg.bottleneck,
                                   attr="cohort_adapter")
                load_level_state_dict(tmpl, pre_state, attr="cohort_adapter")
                pre_patient_aucs[hid] = eval_auc(tmpl, traj_lds[hid], label_fn, device).auc
                del tmpl
        history.append(log)
        # Test AUROC after this round, for the BWT trajectory. Hospitals with no
        # test windows (possible at low alpha) are skipped.
        round_aucs: dict[int, float | None] = {}
        for hid, state in coord.cohort_states.items():
            if hid not in traj_lds:
                round_aucs[hid] = None
                continue
            tmpl = copy.deepcopy(population_clf)
            attach_ffn_adapter(tmpl.backbone,
                               bottleneck=coord_cfg.cohort_cfg.bottleneck,
                               attr="cohort_adapter")
            load_level_state_dict(tmpl, state, attr="cohort_adapter")
            round_aucs[hid] = eval_auc(tmpl, traj_lds[hid], label_fn, device).auc
            del tmpl
        per_round_cohort_aucs.append(round_aucs)
        # Gain from patient feedback, per hospital.
        propagation_gains.append({
            hid: (round_aucs[hid] - pre_patient_aucs[hid])
                  if (round_aucs[hid] is not None and pre_patient_aucs.get(hid) is not None)
                  else None
            for hid in round_aucs
        })
        # Score every round on the probe slice. Round 0 is the population model, so
        # this shows whether any round beats not adapting at all, which tells a
        # noisy selection apart from a real null result. The probe never drives
        # selection, so reading it keeps the test set untouched.
        if probe_loaders:
            _pr = {}
            for hid, state in coord.cohort_states.items():
                if hid not in probe_loaders:
                    continue
                _t = copy.deepcopy(population_clf)
                attach_ffn_adapter(_t.backbone,
                                   bottleneck=coord_cfg.cohort_cfg.bottleneck,
                                   attr="cohort_adapter")
                load_level_state_dict(_t, state, attr="cohort_adapter")
                _pr[hid] = eval_auc(_t, probe_loaders[hid], label_fn, device).auc
                del _t
            probe_round_aucs.append(macro_average(_pr.values()))

        # Population trajectory (axes 3 and 4). The population adapter has the same
        # shape as the cohort adapter and loads the same way. Before the first sync
        # there is no adapter, so the population model is still the frozen baseline.
        if pop_eval_loader is not None:
            _ps = coord.population_adapter_state
            if _ps is None:
                population_round_aucs.append(pop_baseline_auc)
            else:
                _t = copy.deepcopy(population_clf)
                attach_ffn_adapter(_t.backbone,
                                   bottleneck=coord_cfg.cohort_cfg.bottleneck,
                                   attr="cohort_adapter")
                load_level_state_dict(_t, _ps, attr="cohort_adapter")
                population_round_aucs.append(
                    eval_auc(_t, pop_eval_loader, label_fn, device).auc)
                del _t

        # Validation macro AUROC, used for round selection (kept apart from test).
        val_aucs: dict[int, float | None] = {}
        for hid, state in coord.cohort_states.items():
            if hid not in val_lds:
                continue
            tmpl = copy.deepcopy(population_clf)
            attach_ffn_adapter(tmpl.backbone,
                               bottleneck=coord_cfg.cohort_cfg.bottleneck,
                               attr="cohort_adapter")
            load_level_state_dict(tmpl, state, attr="cohort_adapter")
            val_aucs[hid] = eval_auc(tmpl, val_lds[hid], label_fn, device).auc
            del tmpl
        _val_macro = macro_average(val_aucs.values())
        if _val_macro is not None and (_best_val is None or _val_macro > _best_val):
            _best_val = _val_macro
            _best_states = {h: {k: v.clone() for k, v in st.items()}
                            for h, st in coord.cohort_states.items()}
            _best_round = r
        # Also track each centre's own best round, for per_hospital_selection (see
        # baselines._round_loop_with_early_stopping).
        for _h, _a in val_aucs.items():
            if _a is None:
                continue
            if _per_hosp_best.get(_h) is None or _a > _per_hosp_best[_h]:
                _per_hosp_best[_h] = _a
                _per_hosp_round[_h] = r
                _per_hosp_state[_h] = {k: v.clone()
                                       for k, v in coord.cohort_states[_h].items()}
        if monitor.update(r, _val_macro):
            break
    wall_clock = time.perf_counter() - t0

    # Restore the best validation round. Round 0 (the identity adapter, i.e. the
    # population model) is a candidate, so the selected model is never worse on
    # validation than where it started. With patience=999 early stopping never
    # fires, so this is the only model selection there is.
    from .baselines import resolve_selection, state_delta_norm
    _sel = resolve_selection(
        per_hospital_selection=per_hospital_selection,
        global_states=_best_states, global_val=_best_val, global_round=_best_round,
        per_hosp_state=_per_hosp_state, per_hosp_best=_per_hosp_best,
        per_hosp_round=_per_hosp_round,
    )
    _best_states, _best_val, _best_round = _sel.states, _sel.val_auroc, _sel.round
    _delta_norm = state_delta_norm(_best_states, _init_states)
    print(f"    model selection: best val macro AUROC {_best_val:.4f} @ round "
          f"{'per-centre' if _best_round is None else _best_round}"
          f" (|Δ| from round 0 = {_delta_norm:.3e})"
          if _best_val is not None else "    model selection: val undefined")
    coord.cohort_states = _best_states

    # Probe-slice score of the selected model. It is used to compare configurations
    # and never to pick a round (see MethodResult.probe_auroc).
    _probe_auroc = None
    if probe_loaders:
        _pv = {}
        for hid, state in coord.cohort_states.items():
            if hid not in probe_loaders:
                continue
            _t = copy.deepcopy(population_clf)
            attach_ffn_adapter(_t.backbone,
                               bottleneck=coord_cfg.cohort_cfg.bottleneck,
                               attr="cohort_adapter")
            load_level_state_dict(_t, state, attr="cohort_adapter")
            _pv[hid] = eval_auc(_t, probe_loaders[hid], label_fn, device).auc
            del _t
        _probe_auroc = macro_average(_pv.values())

    # Forward transfer (axis 3): each cohort model, and the frozen population model,
    # scored on the same pooled reference set drawn from all centres. Scoring on the
    # centre's own test partition would just be Delta_cohort again.
    fwt_global: dict[int, float | None] = {}
    if pop_eval_loader is not None and pop_baseline_auc is not None:
        for hid, state in coord.cohort_states.items():
            _t = copy.deepcopy(population_clf)
            attach_ffn_adapter(_t.backbone,
                               bottleneck=coord_cfg.cohort_cfg.bottleneck,
                               attr="cohort_adapter")
            load_level_state_dict(_t, state, attr="cohort_adapter")
            a = eval_auc(_t, pop_eval_loader, label_fn, device).auc
            fwt_global[hid] = None if a is None else a - pop_baseline_auc
            del _t
        print(f"    FWT (cohort model on the GLOBAL held-out set, vs frozen "
              f"population): {({h: None if v is None else round(v, 4) for h, v in fwt_global.items()})}")

    # Full test metrics: AUROC, AUPRC and ECE.
    from ..personalisation.evaluate import eval_full
    cohort_metrics: dict[int, dict] = {}
    for hid, state in coord.cohort_states.items():
        if hid not in eval_loaders:
            cohort_metrics[hid] = {"auc": None, "auprc": None, "ece": None,
                                   "n": None, "n_pos": None}
            continue
        tmpl = copy.deepcopy(population_clf)
        attach_ffn_adapter(tmpl.backbone,
                           bottleneck=coord_cfg.cohort_cfg.bottleneck,
                           attr="cohort_adapter")
        load_level_state_dict(tmpl, state, attr="cohort_adapter")
        m = eval_full(tmpl, eval_loaders[hid], label_fn, device)
        cohort_metrics[hid] = {"auc": m.auroc, "auprc": m.auprc, "ece": m.ece,
                               "n": m.n, "n_pos": m.n_pos,
                               "preds": m.preds, "labels": m.labels}
        del tmpl

    # Patient level. Predictions are pooled across up to ``patient_eval_n`` patients
    # per hospital, since one patient's slice is often single-class. This uses the
    # same helper as local_finetune, so both are judged by the same accept/reject
    # rule.
    from .baselines import evaluate_patient_layer

    # How many patients per hospital go into the pooled patient AUROC. If unset, it
    # falls back to n_iters // 10 + 1, which is a weak default; the reported runs
    # set patient_eval_patients explicitly.
    if patient_eval_n is None:
        patient_eval_n = max(1, int(getattr(coord_cfg, "patient_cfg").n_iters // 10) + 1)
    patient_eval_n = max(1, int(patient_eval_n))

    _pl = evaluate_patient_layer(
        population_clf, coord.cohort_states, hospital_patient_streams,
        label_fn, device, coord_cfg, patient_eval_n,
        patient_min_pos=patient_min_pos,
    )
    patient_aucs = _pl["aucs"]
    patient_cohort_ref = _pl["cohort_aucs"]
    patient_accepted = _pl["accepted"]
    patient_considered = _pl["considered"]
    patient_unevaluable = _pl["unevaluable"]
    patient_pooled_pos = _pl["pooled_pos"]
    patient_records = _pl["records"]

    return MethodResult(
        aucs={hid: m["auc"] for hid, m in cohort_metrics.items()},
        auprcs={hid: m["auprc"] for hid, m in cohort_metrics.items()},
        eces={hid: m["ece"] for hid, m in cohort_metrics.items()},
        ns={hid: m["n"] for hid, m in cohort_metrics.items()},
        n_pos={hid: m["n_pos"] for hid, m in cohort_metrics.items()},
        n_rounds=monitor.last_round,
        selected_round=_best_round,
        selected_delta_norm=_delta_norm,
        val_auroc=_best_val,
        val_auroc_round0=macro_average(_val0.values()),
        probe_auroc=_probe_auroc,
        wall_clock_sec=wall_clock,
        val_history=monitor.history,
        extra={
            "selected_round_per_hospital": _sel.per_hospital_round,
            "test_preds": {h: m["preds"] for h, m in cohort_metrics.items()},
            "test_labels": {h: m["labels"] for h, m in cohort_metrics.items()},
            "patient_aucs": patient_aucs,
            "patient_accepted": patient_accepted,
            "patient_considered": patient_considered,
            "patient_unevaluable": patient_unevaluable,
            "patient_pooled_pos": patient_pooled_pos,
            "patient_cohort_ref": patient_cohort_ref,
            "patient_records": patient_records,
            "cohort_state_params": parameters_updated(
                next(iter(coord.cohort_states.values()), None)),
            "population_state_params": parameters_updated(
                coord.population_adapter_state),
            "fwt_global": fwt_global,
            "probe_round_aucs": probe_round_aucs,
            "population_round_aucs": population_round_aucs,
            "population_baseline_auc": pop_baseline_auc,
            "per_round_cohort_aucs": per_round_cohort_aucs,
            "propagation_gains": propagation_gains,
            "coordinator_history": history,
        },
    )


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = Path(cfg.train_dir)
    test_dir = Path(cfg.input_dir)
    model_path = Path(cfg.model_path)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the resolved config next to the results. inference.yaml turns off Hydra's
    # own output folder, so without this a finished cell would not record what
    # produced it.
    dump_config(cfg, output_dir / "config.yaml")

    # Settings
    num_hospitals = int(getattr(cfg, "num_hospitals", 3))
    # alphas can be a Hydra list (``+alphas=[0.1,1.0]``), a Python list, or a comma-
    # separated string.
    alphas_raw = getattr(cfg, "alphas", "0.1,1.0,10.0")
    try:
        # ListConfig / list iteration
        alphas = [float(a) for a in alphas_raw]
    except TypeError:
        alphas = [float(a) for a in str(alphas_raw).split(",")]
    # Rounds: max_rounds plus early stopping (patience, min_rounds).
    max_rounds = int(getattr(cfg, "max_rounds", getattr(cfg, "rounds", 15)))
    patience = int(getattr(cfg, "patience", 3))
    min_rounds = int(getattr(cfg, "min_rounds", 2))
    patients_per_round = int(getattr(cfg, "patients_per_round", 2))
    max_patients_per_hospital = int(getattr(cfg, "max_patients_per_hospital", 3))
    n_patient_iters = int(getattr(cfg, "n_patient_iters", 5))
    cohort_bottleneck = int(getattr(cfg, "cohort_bottleneck", 64))
    patient_bottleneck = cohort_bottleneck  # must match for upward prop
    cohort_local_steps = int(getattr(cfg, "cohort_local_steps", 5))
    # WP4 tasks 1-2: the cohort adapts at a medium learning rate and patient
    # adapters at a faster one.
    cohort_lr = float(getattr(cfg, "cohort_lr", 1e-3))
    patient_lr = float(getattr(cfg, "patient_lr", 5e-3))
    n_pop_eval = int(getattr(cfg, "n_pop_eval", 50))
    # 0 = score the whole test/val partition. The small default is for smoke runs.
    eval_per_hosp = int(getattr(cfg, "eval_per_hospital", 200))
    val_per_hosp = int(getattr(cfg, "val_per_hospital", eval_per_hosp))
    # The per-round trajectory costs two extra passes per round, so it can stay
    # capped even when the reported test set is not.
    traj_per_hosp = int(getattr(cfg, "traj_eval_per_hospital", eval_per_hosp))
    # Optionally size the validation set by the number of positives it will hold,
    # not by a window count. At 2 % prevalence (mortality) 2,000 windows hold only
    # about 42 positives, which makes round selection noisy; readmission at 20.6 %
    # gets about 410 from the same count. 0 turns this off and uses
    # val_per_hospital.
    val_target_pos = int(getattr(cfg, "val_target_pos", 0))
    # Fraction of each centre's validation split held back as a probe slice. The
    # probe is never used to pick a round, so it can compare configurations without
    # touching test. 0 turns it off.
    val_probe_frac = float(getattr(cfg, "val_probe_frac", 0.0))
    # Evidence floor for the MACRO average: a hospital needs at least this many
    # positives and negatives. At low alpha a centre can be tiny (69 patients at
    # alpha=0.1) and its AUROC is close to a coin flip, yet it would count as much
    # as a large centre. Such hospitals still get their own row, flagged, but are
    # left out of MACRO. Set both to 0 to turn the floor off.
    min_eval_pos = int(getattr(cfg, "min_eval_pos", 10))
    min_eval_neg = int(getattr(cfg, "min_eval_neg", 10))
    # Eval batch size. Predictions are the same as at batch 1 (see
    # evaluation/batching.py); this only makes evaluation faster.
    eval_batch_size = int(getattr(cfg, "eval_batch_size", 1))
    # Cohort training batch size. Unlike the eval batch this does change training:
    # cohort_local_steps counts optimiser steps, so a centre sees steps x batch
    # windows per round.
    cohort_batch_size = int(getattr(cfg, "cohort_batch_size", 1))
    # Whether standard_fedavg trains with the same continual-learning regulariser as
    # the other methods. The TD defines the baseline as FedAvg "without continual
    # learning or hierarchy", so the default is False. True gives a CL-matched
    # variant, which is an ablation, not the baseline.
    fedavg_use_cl = bool(getattr(cfg, "fedavg_use_cl", False))
    # patient_batch_size: batch size for patient adapters. At batch 1 with few
    # iterations an adapter sees only a handful of windows and memorises them.
    #
    # cohort_balanced_sampling: sample cohort training windows the way the
    # population head was fitted (keep every positive, cap negatives per positive).
    # Without it the adapter trains on the natural ~2 % positive rate while the head
    # was fitted on about 20 %. pos_weight is recomputed from the resampled draw so
    # the correction is not applied twice.
    patient_batch_size = int(getattr(cfg, "patient_batch_size", 8))
    cohort_balanced_sampling = bool(getattr(cfg, "cohort_balanced_sampling", False))
    cohort_max_neg_per_pos = float(getattr(cfg, "cohort_max_neg_per_pos", 3.0))
    # Patient-level sample size: a patient needs patient_min_samples windows to
    # qualify, and half of them become evaluation windows.
    patient_min_samples = int(getattr(cfg, "patient_min_samples", 4))
    patient_eval_patients = getattr(cfg, "patient_eval_patients", None)
    if patient_eval_patients is not None:
        patient_eval_patients = int(patient_eval_patients)
    eta_cohort = float(getattr(cfg, "eta_cohort", 0.3))
    eta_population = float(getattr(cfg, "eta_population", 0.1))
    seed = int(getattr(cfg, "seed", 42))
    # Two separate seeds. ``seed`` fixes the data: the Dirichlet partition, the
    # balanced-sampling draw and the dataloader order. ``run_seed`` fixes training:
    # torch's global RNG, which initialises each adapter.
    #
    # Keep ``seed`` fixed across a campaign so every cell uses the same partition,
    # and vary ``run_seed`` to measure run-to-run variance on that partition.
    # Changing ``seed`` would also redraw which patients belong to which centre.
    run_seed = int(getattr(cfg, "run_seed", seed))
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    print(f"  seeds: data seed={seed}, run seed={run_seed}")
    tasks_filter = getattr(cfg, "tasks", None)
    # With ``splits_dir`` set, the pre-computed train/val/test partitions are loaded
    # from JSON; otherwise a Dirichlet split is drawn inline.
    splits_dir = getattr(cfg, "splits_dir", None)
    # Label the partition used ("classaware" or "random"), taken from the splits
    # path, and keep the path itself too, so every results row records its own data
    # split.
    _sd = str(splits_dir) if splits_dir else ""
    split_label = (getattr(cfg, "split_scheme", None)
                   or ("classaware" if "classaware" in _sd
                       else "random" if _sd else None))

    Vocabulary.from_path(test_dir)
    with open(test_dir / STATIC_DATA_FN, "rb") as f:
        test_static = pickle.load(f)
    test_cat, test_num = build_static_arrays(test_static)
    # Placeholder collate. The real one is built per task inside the loop, because
    # it needs the dataset to recover each window's real inter-event times.
    collate_fn = make_collate(test_cat, test_num)

    # The population classifier is built per task because its head is task-specific.
    # Without ``population_head_dir`` the head is random and population_only scores
    # at chance.
    population_head_dir = getattr(cfg, "population_head_dir", None)
    if population_head_dir is None:
        print("WARNING: no population_head_dir — the classifier head will be "
              "RANDOMLY INITIALISED, making population_only a chance-level "
              "baseline and delta_cohort uninterpretable.")
    all_pids = list(test_static.keys())

    selected_tasks = {k: v for k, v in TASKS.items()}
    if tasks_filter is not None:
        if isinstance(tasks_filter, str):
            keep = set(s.strip() for s in tasks_filter.split(","))
        else:
            keep = set(str(s).strip() for s in tasks_filter)
        selected_tasks = {k: v for k, v in TASKS.items() if k in keep}
    print(f"tasks={list(selected_tasks)}, alphas={alphas}")

    # Continual-learning settings per level (WP4 task 4). "distill" adds a KD term
    # towards the level above: the population model for the cohort, the cohort model
    # for patients.
    cohort_cl_strategy = str(getattr(cfg, "cohort_cl_strategy", "none"))
    cohort_cl_alpha = float(getattr(cfg, "cohort_cl_alpha", 0.3))
    cohort_cl_temperature = float(getattr(cfg, "cohort_cl_temperature", 2.0))
    patient_cl_strategy = str(getattr(cfg, "patient_cl_strategy", "none"))
    patient_cl_alpha = float(getattr(cfg, "patient_cl_alpha", 0.3))
    patient_cl_temperature = float(getattr(cfg, "patient_cl_temperature", 2.0))

    # Which methods to run (default: all five). The WP4 ablations only change upward
    # propagation, so population_only, local_finetune and standard_fedavg would give
    # identical numbers in every arm; skipping them saves time. A skipped method
    # leaves its columns empty.
    ALL_METHODS = ("population_only", "local_finetune", "standard_fedavg",
                   "top_down_only", "hierarchical")
    methods_cfg = getattr(cfg, "methods", None)
    if methods_cfg is None:
        methods = set(ALL_METHODS)
    else:
        if isinstance(methods_cfg, str):
            methods_cfg = [m.strip() for m in methods_cfg.split(",") if m.strip()]
        methods = {str(m) for m in methods_cfg}
        unknown = methods - set(ALL_METHODS)
        if unknown:
            raise ValueError(
                f"unknown method(s) {sorted(unknown)}; valid: {list(ALL_METHODS)}"
            )
    print(f"methods: {[m for m in ALL_METHODS if m in methods]}")

    weighting_scheme = str(getattr(cfg, "weighting_scheme", "volume"))
    _valid_schemes = {"volume", "uniform", "recency", "uncertainty"}
    if weighting_scheme not in _valid_schemes:
        raise ValueError(
            f"weighting_scheme must be one of {sorted(_valid_schemes)}, "
            f"got {weighting_scheme!r}"
        )
    # Upward weighting (WP4 task 3), optionally set per level. Unset means "use
    # weighting_scheme at that level".
    patient_weighting_scheme = getattr(cfg, "patient_weighting_scheme", None)
    cohort_weighting_scheme = getattr(cfg, "cohort_weighting_scheme", None)
    for _name, _v in (("patient_weighting_scheme", patient_weighting_scheme),
                      ("cohort_weighting_scheme", cohort_weighting_scheme)):
        if _v is not None and str(_v) not in _valid_schemes:
            raise ValueError(
                f"{_name} must be one of {sorted(_valid_schemes)}, got {_v!r}")
    patient_weighting_scheme = (str(patient_weighting_scheme)
                                if patient_weighting_scheme is not None else None)
    cohort_weighting_scheme = (str(cohort_weighting_scheme)
                               if cohort_weighting_scheme is not None else None)
    recency_half_life_days = float(getattr(cfg, "recency_half_life_days", 180.0))
    # Method changes, all off by default:
    # - each centre keeps its own best round instead of one global round;
    per_hospital_selection = bool(getattr(cfg, "per_hospital_selection", False))
    # - unfreeze only the classifier head's intercepts, so a per-centre base-rate
    # shift is corrected in the head rather than through the adapter;
    train_head_bias_only = bool(getattr(cfg, "train_head_bias_only", False))
    # - train the whole classifier head per centre. The cross-centre transfer test
    # (scripts/cross_centre_transfer.py) suggested capacity in the head matters more
    # than capacity in the adapter; this tests that.
    #
    # A cell run with any of these on is not comparable to the reported grid.
    train_head = bool(getattr(cfg, "train_head", False))
    # Run the federated steps through Flower (an approved change to TD 3.5.1). If
    # Flower is requested but not installed, fail instead of silently falling back
    # to the in-process path, so a result is never credited to the wrong runtime.
    use_flower = bool(getattr(cfg, "use_flower", False))
    if use_flower:
        try:
            import flwr as _flwr
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "use_flower=true but flwr is not importable. Refusing to fall "
                "back to the in-process path: the run would silently produce a "
                "result attributed to the wrong pipeline. Install the extra "
                "(pip install -e '.[federated]') or unset USE_FLOWER."
            ) from exc
        print(f"  federated runtime: FLOWER {_flwr.__version__} "
              f"(client/server; cohort<->population crosses the boundary, "
              f"patient level is local by construction)")
    else:
        print("  federated runtime: in-process simulation (single process)")
    # TD baseline: local fine-tuning is "per-centre AND per-patient fine-tuning
    # without coordination", so the patient level is on by default. It roughly
    # triples that method's runtime; set false only to reproduce older per-centre-
    # only runs.
    local_finetune_patients = bool(getattr(cfg, "local_finetune_patients", True))
    # WP4 task 5: trigger a population update after N patient updates. This differs
    # from the round trigger because one round holds many patient updates.
    population_update_after_n_patients = getattr(
        cfg, "population_update_after_n_patients", None)
    if population_update_after_n_patients is not None:
        population_update_after_n_patients = int(population_update_after_n_patients)

    # WP4 task 5: coordination protocol. A population update fires every
    # ``population_update_freq`` rounds, or when a cohort drifts further than
    # ``divergence_threshold`` (L2) from the last population state. None disables
    # the divergence trigger; the reported runs use the frequency trigger only.
    divergence_threshold = getattr(cfg, "divergence_threshold", None)
    if divergence_threshold is not None:
        divergence_threshold = float(divergence_threshold)
    population_update_freq = int(
        getattr(cfg, "population_update_freq", max(2, max_rounds))
    )

    coord_cfg_template = CoordinatorConfig(
        cohort_cfg=CohortConfig(
            bottleneck=cohort_bottleneck, lr=cohort_lr,
            local_epochs=1, local_steps_per_epoch=cohort_local_steps,
            objective="downstream",
            cl_strategy=cohort_cl_strategy, cl_alpha=cohort_cl_alpha,
            cl_temperature=cohort_cl_temperature,
            train_head_bias_only=train_head_bias_only,
            train_head=train_head,
        ),
        patient_cfg=PatientConfig(
            bottleneck=patient_bottleneck, lr=patient_lr,
            n_iters=n_patient_iters,
            cl_strategy=patient_cl_strategy, cl_alpha=patient_cl_alpha,
            cl_temperature=patient_cl_temperature,
        ),
        patients_per_round=patients_per_round,
        population_update_freq=population_update_freq,
        weighting_scheme=weighting_scheme,
        patient_weighting_scheme=patient_weighting_scheme,
        cohort_weighting_scheme=cohort_weighting_scheme,
        recency_half_life_days=recency_half_life_days,
        divergence_threshold=divergence_threshold,
        population_update_after_n_patients=population_update_after_n_patients,
        eta_cohort=eta_cohort, eta_population=eta_population,
        # Top-down refresh: blend the cohort towards the population state instead of
        # overwriting it (1.0 = overwrite).
        refresh_blend=float(getattr(cfg, "refresh_blend", 0.3)),
        patient_rotation=bool(getattr(cfg, "patient_rotation", False)),
        # Run the federated steps through Flower's client/server components. The
        # maths is the same as the in-process loop (see
        # tests/test_flower_equivalence.py), but the order differs: a Flower client
        # finishes its cohort and patient work in one fit() call. That changes the
        # order of random draws, so results agree within about +/-0.0008 AUROC
        # rather than exactly. Do not mix the two in one grid.
        use_flower=use_flower,
    )
    def _cap_str(n):
        return "ALL" if not n else str(n)
    print(f"eval sizes/hospital: test={_cap_str(eval_per_hosp)} val={_cap_str(val_per_hosp)} "
          f"traj={_cap_str(traj_per_hosp)} | eval_batch_size={eval_batch_size} "
          f"cohort_batch_size={cohort_batch_size}")
    print(f"WP4 upward prop (task 3): patient->cohort="
          f"{patient_weighting_scheme or weighting_scheme}, cohort->population="
          f"{cohort_weighting_scheme or weighting_scheme}"
          f", recency half-life={recency_half_life_days:g}d")
    print(f"WP4 triggering (task 5): population_update_freq={population_update_freq} rounds"
          f" | after_n_patients={population_update_after_n_patients}"
          f" | divergence_threshold={divergence_threshold}"
          f"  (which one fired is logged per round as update_trigger)")
    print(f"patient layer: min_samples={patient_min_samples} "
          f"(≈{patient_min_samples // 2} eval windows/patient), "
          f"max/hospital={max_patients_per_hospital}, per_round={patients_per_round}, "
          f"rotation={bool(getattr(cfg, 'patient_rotation', False))}, "
          f"pooled_for_auc={patient_eval_patients if patient_eval_patients else 'auto(n_iters//10+1)'}")
    print(f"WP4 lrs: cohort_lr={cohort_lr} (medium), patient_lr={patient_lr} (fast)")
    print(f"cohort CL: {cohort_cl_strategy} (α={cohort_cl_alpha}, T={cohort_cl_temperature})")
    print(f"patient CL: {patient_cl_strategy} (α={patient_cl_alpha}, T={patient_cl_temperature})")

    rows: list[dict] = []
    patient_rows: list[dict] = []
    for task_name, dataset_cls in selected_tasks.items():
        test_ds = dataset_cls(test_dir)
        # Rebuild the collate with the dataset so the T-LSTM gets real time gaps.
        # With a constant gap the time decay is stuck at 0.7615 (see
        # cocofl.data.elapsed).
        collate_fn = make_collate(test_cat, test_num, dataset=test_ds)
        label_fn = make_label_fn(task_name)
        population_clf = build_population_classifier(
            model_path, device,
            head_path=(Path(population_head_dir) / f"{task_name}.pt")
            if population_head_dir else None,
        )
        # Class weight for cohort and patient training. The population head was
        # fitted with a weighted loss; training here without one pulls predictions
        # back towards the base rate. One weight per task is enough because positive
        # rates are similar across centres.
        from ..downstream.probe import stratified_subsample, window_labels
        _y_task = window_labels(
            test_ds, POSITIVE_TOKENS[task_name], horizon_microseconds(task_name),
        )
        _prev = float(_y_task.mean())
        task_pos_weight = ((1.0 - _prev) / _prev) if _prev > 0 else None
        print(f"  task prevalence {_prev:.3%} -> cohort/patient pos_weight="
              f"{task_pos_weight:.1f}" if task_pos_weight else "  no positives")
        # Per-task validation size, so every task's round selection rests on a
        # similar number of positives.
        val_per_hosp_task = val_per_hosp
        if val_target_pos > 0 and _prev > 0:
            needed = int(-(-val_target_pos // _prev))  # ceil
            val_per_hosp_task = max(val_per_hosp, needed)
            print(f"  val sizing: target {val_target_pos} positives at "
                  f"{_prev:.3%} prevalence -> val_per_hospital="
                  f"{val_per_hosp_task} (was {val_per_hosp})")
        # Used by every method, so the weighting is the same for all of them.
        coord_cfg_template.cohort_cfg.pos_weight = task_pos_weight
        coord_cfg_template.patient_cfg.pos_weight = task_pos_weight

        for alpha in alphas:
            print(f"\n=== task={task_name}  α={alpha} ===")
            hospital_val_loaders: dict = {}
            if splits_dir is not None:
                # Pre-computed fixed split (WP2 task 4: reproducible, no leakage).
                from ..federated import load_split, as_hospital_split
                split_data = load_split(splits_dir, alpha)
                train_split = as_hospital_split(split_data, partition="train")
                val_split = as_hospital_split(split_data, partition="val")
                test_split = as_hospital_split(split_data, partition="test")
                for s in (train_split, val_split, test_split):
                    s.bind_to_dataset(test_ds)
                _train_idx = {hid: list(idxs)
                              for hid, idxs in train_split.indices_per_hospital.items()
                              if idxs}
                if cohort_balanced_sampling:
                    _rng_bal = np.random.default_rng(seed)
                    _before = {h: len(v) for h, v in _train_idx.items()}
                    _train_idx = {
                        hid: stratified_subsample(
                            _y_task, idxs, 0, _rng_bal,
                            max_neg_per_pos=cohort_max_neg_per_pos,
                        )
                        for hid, idxs in _train_idx.items()
                    }
                    _pooled = np.concatenate(
                        [np.asarray(v, dtype=np.int64) for v in _train_idx.values()]
                    ) if _train_idx else np.array([], dtype=np.int64)
                    _p2 = float(_y_task[_pooled].mean()) if _pooled.size else 0.0
                    task_pos_weight = ((1.0 - _p2) / _p2) if _p2 > 0 else None
                    coord_cfg_template.cohort_cfg.pos_weight = task_pos_weight
                    coord_cfg_template.patient_cfg.pos_weight = task_pos_weight
                    print(f"  balanced cohort sampling: "
                          f"{ {h: (_before[h], len(v)) for h, v in _train_idx.items()} } "
                          f"-> prevalence {_p2:.2%}, pos_weight="
                          f"{task_pos_weight:.1f}")
                hospital_train_loaders = {
                    hid: build_hospital_train_loader(
                        test_ds, idxs, collate_fn,
                        batch_size=cohort_batch_size, seed=seed,
                    )
                    for hid, idxs in _train_idx.items()
                }
                # Split each centre's validation set into a selection part and a
                # probe part. The probe never drives round selection, so it can
                # compare configurations while the test set stays untouched.
                _val_sel_idx, _val_probe_idx = {}, {}
                for hid, idxs in val_split.indices_per_hospital.items():
                    if not idxs:
                        continue
                    capped = _cap(idxs, val_per_hosp_task)
                    if val_probe_frac > 0.0:
                        n_probe = int(len(capped) * val_probe_frac)
                        # Interleave instead of slicing: the indices are time-
                        # ordered, so a tail slice would make the probe a later time
                        # period than the selection part.
                        _val_probe_idx[hid] = capped[0::2][:n_probe] if n_probe else []
                        _probe_set = set(_val_probe_idx[hid])
                        _val_sel_idx[hid] = [i for i in capped if i not in _probe_set]
                    else:
                        _val_sel_idx[hid] = capped
                hospital_val_loaders = {
                    hid: make_eval_loader(test_ds, idxs, collate_fn,
                                           batch_size=eval_batch_size)
                    for hid, idxs in _val_sel_idx.items() if idxs
                }
                hospital_probe_loaders = {
                    hid: make_eval_loader(test_ds, idxs, collate_fn,
                                           batch_size=eval_batch_size)
                    for hid, idxs in _val_probe_idx.items() if idxs
                }
                if hospital_probe_loaders:
                    print(f"  config probe slice: "
                          f"{ {h: len(v) for h, v in _val_probe_idx.items()} } windows "
                          f"(val selection keeps "
                          f"{ {h: len(v) for h, v in _val_sel_idx.items()} })")
                hospital_eval_loaders = {
                    hid: make_eval_loader(test_ds, _cap(idxs, eval_per_hosp), collate_fn,
                                           batch_size=eval_batch_size)
                    for hid, idxs in test_split.indices_per_hospital.items() if idxs
                }
                # Trajectory set: a prefix of the same test indices, so BWT compares
                # the same windows across rounds.
                hospital_traj_loaders = {
                    hid: make_eval_loader(test_ds, _cap(idxs, traj_per_hosp), collate_fn,
                                           batch_size=eval_batch_size)
                    for hid, idxs in test_split.indices_per_hospital.items() if idxs
                }
                # Patient streams come from the train partition: these patients get
                # their own adapters, and their updates can flow up to the cohort.
                hospital_patient_streams = {
                    hid: build_patient_streams(
                        test_ds, train_split.pids_per_hospital[hid], collate_fn,
                        max_patients=max_patients_per_hospital,
                        min_samples=patient_min_samples,
                        eval_batch_size=eval_batch_size,
                        patient_batch_size=patient_batch_size,
                    ) for hid in hospital_train_loaders
                }
            else:
                split = dirichlet_split_patients(all_pids, num_hospitals, alpha, seed=seed)
                split.bind_to_dataset(test_ds)
                hospital_train_loaders, _inline_eval_idx = {}, {}
                for hid, indices in split.indices_per_hospital.items():
                    if not indices:
                        continue
                    idxs = list(indices)
                    random.Random(seed).shuffle(idxs)
                    # Hold out a separate eval slice so no method is scored on the
                    # windows it trained on. The shipped scripts always pass
                    # splits_dir, so this path is for ad hoc runs only.
                    n_eval = max(1, int(len(idxs) * 0.3))
                    _inline_eval_idx[hid], train_idxs = idxs[:n_eval], idxs[n_eval:]
                    if not train_idxs:
                        train_idxs = _inline_eval_idx[hid]
                    n_train = 0 if not eval_per_hosp else max(20, eval_per_hosp * 2)
                    hospital_train_loaders[hid] = build_hospital_train_loader(
                        test_ds, _cap(train_idxs, n_train), collate_fn,
                        batch_size=cohort_batch_size, seed=seed,
                    )
                hospital_eval_loaders = {
                    hid: make_eval_loader(test_ds, _cap(idxs, eval_per_hosp),
                                           collate_fn, batch_size=eval_batch_size)
                    for hid, idxs in _inline_eval_idx.items() if idxs
                }
                # No separate validation split here, so early stopping uses the eval
                # loaders. Fine for smoke tests only.
                hospital_val_loaders = hospital_eval_loaders
                hospital_probe_loaders = {}
                hospital_traj_loaders = hospital_eval_loaders
                hospital_patient_streams = {
                    hid: build_patient_streams(
                        test_ds, split.pids_per_hospital[hid], collate_fn,
                        max_patients=max_patients_per_hospital,
                        min_samples=patient_min_samples,
                        eval_batch_size=eval_batch_size,
                        patient_batch_size=patient_batch_size,
                    ) for hid in hospital_train_loaders
                }

            # A skipped method keeps an empty MethodResult, so its columns stay
            # blank.
            pop_res = ft_res = fed_res = td_res = hier_res = MethodResult()
            pop_baseline = None
            # Reference set for the population level (axes 3 and 4). Built in every
            # cell, because the WP4 ablations run hierarchical alone and still need
            # it.
            pop_eval_loader = _build_pop_eval_loader(
                hospital_eval_loaders, test_ds, collate_fn, n_pop_eval,
                batch_size=eval_batch_size,
            )
            pop_baseline = eval_auc(population_clf, pop_eval_loader,
                                    label_fn, device).auc
            cohort_bytes = patient_bytes = naive_bytes = savings = None

            # Method A: Population-only
            if "population_only" in methods:
                pop_res = eval_population_only(population_clf, hospital_eval_loaders,
                                                label_fn, device)
                print(f"  population_only AUROC: {pop_res.aucs}  baseline={pop_baseline}")

            # Method B: Local fine-tune
            if "local_finetune" in methods:
                ft_res = run_local_finetune(
                    population_clf, hospital_train_loaders, hospital_val_loaders,
                    hospital_eval_loaders, label_fn, device, coord_cfg_template,
                    max_rounds=max_rounds, patience=patience, min_rounds=min_rounds,
                    per_hospital_selection=per_hospital_selection,
                    probe_loaders=hospital_probe_loaders,
                    # TD baseline: per-centre and per-patient fine-tuning.
                    patient_streams=(hospital_patient_streams
                                     if local_finetune_patients else None),
                    patient_eval_n=patient_eval_patients,
                    patient_min_pos=min_eval_pos,
                )
                print(f"  local_finetune AUROC: {ft_res.aucs}  (rounds={ft_res.n_rounds}, {ft_res.wall_clock_sec:.0f}s)")

            # Method C: Standard FedAvg
            if "standard_fedavg" in methods:
                fed_res = run_standard_fedavg(
                    population_clf, hospital_train_loaders, hospital_val_loaders,
                    hospital_eval_loaders, label_fn, device, coord_cfg_template,
                    max_rounds=max_rounds, patience=patience, min_rounds=min_rounds,
                    use_cl=fedavg_use_cl, probe_loaders=hospital_probe_loaders,
                )
                from .metrics import (
                    communication_bytes, naive_full_model_bytes,
                    total_communication_cost, comm_savings_ratio,
                )
                # Communication cost per level (axis 6).
                cohort_bytes = communication_bytes(fed_res.extra["server_state"])
                # The patient adapter has the same shape as the cohort adapter.
                patient_bytes = cohort_bytes
                # Naive federation for comparison: every centre sends the whole
                # model every round.
                naive_bytes = naive_full_model_bytes(population_clf)
                savings = comm_savings_ratio(cohort_bytes, naive_bytes)
                print(f"  standard_fedavg AUROC: {fed_res.aucs}  (rounds={fed_res.n_rounds}, "
                      f"{fed_res.wall_clock_sec:.0f}s, comm/round={cohort_bytes}B vs naive "
                      f"{naive_bytes}B → {savings:.1f}x cheaper)")

            # Method D: Top-down only
            if "top_down_only" in methods:
                td_res = run_top_down_only(
                    population_clf, hospital_train_loaders, hospital_val_loaders,
                    hospital_eval_loaders, hospital_patient_streams, label_fn, device,
                    coord_cfg_template,
                    max_rounds=max_rounds, patience=patience, min_rounds=min_rounds,
                    per_hospital_selection=per_hospital_selection,
                    probe_loaders=hospital_probe_loaders,
                )
                print(f"  top_down AUROC: {td_res.aucs}  (rounds={td_res.n_rounds}, {td_res.wall_clock_sec:.0f}s)")

            # Method E: Hierarchical (our method)
            if "hierarchical" in methods:
                hier_res = run_hierarchical_bidir(
                    population_clf, hospital_train_loaders, hospital_patient_streams,
                    label_fn, device, coord_cfg_template,
                    eval_loaders=hospital_eval_loaders,
                    val_loaders=hospital_val_loaders,
                    traj_loaders=hospital_traj_loaders,
                    patient_eval_n=patient_eval_patients,
                    patient_min_pos=min_eval_pos,
                    per_hospital_selection=per_hospital_selection,
                    pop_eval_loader=pop_eval_loader, pop_baseline_auc=pop_baseline,
                    probe_loaders=hospital_probe_loaders,
                    max_rounds=max_rounds, patience=patience, min_rounds=min_rounds,
                )
                print(f"  hierarchical cohort: {hier_res.aucs}  "
                      f"patient: {hier_res.extra.get('patient_aucs', {})}  "
                      f"(rounds={hier_res.n_rounds}, {hier_res.wall_clock_sec:.0f}s)")
            # Frozen population model on the probe slice: the reference for every
            # configuration comparison.
            _probe_pop_auroc = None
            if hospital_probe_loaders:
                _pp = {hid: eval_auc(population_clf, ld, label_fn, device).auc
                       for hid, ld in hospital_probe_loaders.items()}
                _probe_pop_auroc = macro_average(_pp.values())
                print(f"  probe: population_only={_probe_pop_auroc}")

            # Achievability check. Round 0 on the probe slice is the frozen population model,
            # so comparing the best round against it asks: does any round beat not adapting?
            # This is a diagnostic, not a method, since picking a round by the probe is still
            # selection.
            #   gain > 0   a better round exists, so round selection is the bottleneck.
            #   gain <= 0  no round beats not adapting, so the null result is structural.
            _pra = [a for a in hier_res.extra.get("probe_round_aucs", [])
                    if a is not None]
            _probe_best_round = _probe_best_gain = None
            if _pra and _probe_pop_auroc is not None:
                _cands = [_probe_pop_auroc] + _pra      # index 0 == round 0
                _probe_best_round = max(range(len(_cands)), key=lambda i: _cands[i])
                _probe_best_gain = max(_cands) - _probe_pop_auroc
                print(f"  probe achievability: best round={_probe_best_round} "
                      f"gain vs round0={_probe_best_gain:+.4f} "
                      f"({'a better round EXISTS' if _probe_best_gain > 0 else 'NO round beats round 0'})")

            _val_pos_counts = []
            for _h, _ld in sorted(hospital_val_loaders.items()):
                try:
                    _n = sum(int(sum(label_fn(_y))) for _x, _y in _ld)
                except Exception:
                    _n = None
                _val_pos_counts.append(_n)
            print(f"  val positives per hospital: {_val_pos_counts}")

            # Axes 3, 4 and 6
            _pop_traj = hier_res.extra.get("population_round_aucs", []) or []
            _pop_traj_vals = [a for a in _pop_traj if a is not None]
            # Axis 4: population gain from cohort aggregation (the upward half of
            # the loop). Axis 3: the same trajectory shows population stability
            # across rounds.
            population_gain_from_cohorts = (
                (_pop_traj_vals[-1] - pop_baseline)
                if (_pop_traj_vals and pop_baseline is not None) else None
            )
            population_stability_range = (
                (max(_pop_traj_vals) - min(_pop_traj_vals))
                if len(_pop_traj_vals) >= 2 else None
            )
            # Axis 6 asks for parameters per level as well as bytes.
            _params_cohort = (hier_res.extra.get("cohort_state_params")
                              or parameters_updated(fed_res.extra.get("server_state")))
            # Patient adapters have the same shape as cohort adapters (upward
            # propagation requires it).
            _params_patient = _params_cohort
            _params_population = hier_res.extra.get("population_state_params")
            _params_naive = naive_full_model_parameters(population_clf)
            # local_finetune's own patient level: the control for the hierarchical
            # patient layer.
            ft_patient_aucs = ft_res.extra.get("patient_aucs", {})
            ft_patient_ref = ft_res.extra.get("patient_cohort_ref", {})
            hier_patient_aucs = hier_res.extra.get("patient_aucs", {})
            per_round_aucs = hier_res.extra.get("per_round_cohort_aucs", [])
            propagation_gains = hier_res.extra.get("propagation_gains", [])

            # Per-hospital rows
            from .metrics import bwt
            all_hids = set(hospital_train_loaders) | set(hospital_eval_loaders)
            # Eval-set sizes for this cell, for the confidence intervals and the
            # MACRO evidence floor.
            _ns, _n_pos = _eval_sizes(pop_res, hier_res, fed_res, ft_res, td_res)
            macro_hids = _reportable_hospitals(
                sorted(all_hids), _ns, _n_pos, min_eval_pos, min_eval_neg,
            )
            excluded_hids = [h for h in sorted(all_hids) if h not in macro_hids]
            if excluded_hids:
                _sizes_str = ", ".join(
                    f"h{h}: n={_ns.get(h)} pos={_n_pos.get(h)}" for h in excluded_hids
                )
                print(f"  MACRO evidence floor (≥{min_eval_pos} pos, ≥{min_eval_neg} neg): "
                      f"excluded {_sizes_str} | MACRO over hospitals {macro_hids}")
            # Keyed by hospital so the MACRO row can apply the same evidence floor.
            _axis_dc, _axis_dp, _axis_bwt, _axis_pg = {}, {}, {}, {}
            for hid in sorted(all_hids):
                rec = AucPerLevel(
                    task=task_name, hospital=hid,
                    population=pop_res.aucs.get(hid),
                    cohort=hier_res.aucs.get(hid),
                    patient=hier_patient_aucs.get(hid),
                )
                hospital_bwt = bwt([r.get(hid) for r in per_round_aucs])
                # Axis 4: average gain from patient feedback across rounds (after
                # minus before). Positive means patients help the cohort.
                hospital_prop_gain = macro_average([
                    g.get(hid) for g in propagation_gains
                ])
                # Keep the per-hospital axis 2-4 values so the MACRO row can average
                # them.
                _axis_dc[hid] = delta_cohort(rec)
                # Paired: the patient model against the un-adapted model on the same
                # windows. Subtracting the cohort's test-partition AUROC would
                # compare different patients.
                _pc_ref = hier_res.extra.get("patient_cohort_ref", {}).get(hid)
                _p_auc = hier_patient_aucs.get(hid)
                _axis_dp[hid] = (None if (_p_auc is None or _pc_ref is None)
                                 else float(_p_auc) - float(_pc_ref))
                _axis_bwt[hid] = hospital_bwt
                _axis_pg[hid] = hospital_prop_gain
                rows.append({
                    "task": task_name, "alpha": alpha, "hospital": hid,
                    # AUROC per method
                    "auc_population_only": pop_res.aucs.get(hid),
                    "auc_local_finetune": ft_res.aucs.get(hid),
                    "auc_standard_fedavg": fed_res.aucs.get(hid),
                    "auc_top_down_only": td_res.aucs.get(hid),
                    "auc_hierarchical_cohort": hier_res.aucs.get(hid),
                    "auc_hierarchical_patient": hier_patient_aucs.get(hid),
                    # AUPRC per method (axis 1)
                    "auprc_population_only": pop_res.auprcs.get(hid),
                    "auprc_local_finetune": ft_res.auprcs.get(hid),
                    "auprc_standard_fedavg": fed_res.auprcs.get(hid),
                    "auprc_top_down_only": td_res.auprcs.get(hid),
                    "auprc_hierarchical_cohort": hier_res.auprcs.get(hid),
                    # ECE per method (axis 1)
                    "ece_population_only": pop_res.eces.get(hid),
                    "ece_local_finetune": ft_res.eces.get(hid),
                    "ece_standard_fedavg": fed_res.eces.get(hid),
                    "ece_top_down_only": td_res.eces.get(hid),
                    "ece_hierarchical_cohort": hier_res.eces.get(hid),
                    # Personalisation gain and transfer
                    "pop_baseline": pop_baseline,
                    "delta_cohort_hier": delta_cohort(rec),
                    "delta_patient_hier": _axis_dp[hid],
                    "patient_cohort_ref_auroc": hier_res.extra.get(
                        "patient_cohort_ref", {}).get(hid),
                    # The old unpaired definition, kept for comparison.
                    "delta_patient_hier_unpaired": delta_patient(rec),
                    # Same-centre difference, i.e. Delta_cohort rescaled. Not
                    # forward transfer.
                    "fwt_hier_samecentre": fwt(rec, pop_baseline),
                    # Axis 3 forward transfer, on data outside the centre.
                    "fwt_hier_global": hier_res.extra.get("fwt_global", {}).get(hid),
                    "bwt_hier_cohort": hospital_bwt,
                    # Axis 4: cohort gain from patient feedback, averaged over
                    # rounds.
                    "patient_propagation_gain": hospital_prop_gain,
                    # Communication (axis 6): adapter cost per round, the naive
                    # full-model cost, and totals over training.
                    "comm_bytes_cohort_per_round": cohort_bytes,
                    "comm_bytes_patient_per_round": patient_bytes,
                    "comm_naive_full_model_bytes_per_round": naive_bytes,
                    "comm_savings_ratio_vs_naive": savings,
                    # Communication is measured from the FedAvg server state, so it
                    # is blank when standard_fedavg did not run.
                    "comm_total_bytes_fedavg": None if cohort_bytes is None else
                        total_communication_cost(
                            cohort_bytes, num_hospitals, fed_res.n_rounds,
                        ),
                    "comm_total_bytes_hierarchical": None if cohort_bytes is None else
                        total_communication_cost(
                            cohort_bytes + patient_bytes, num_hospitals,
                            hier_res.n_rounds,
                        ),
                    "comm_total_bytes_naive": None if naive_bytes is None else
                        total_communication_cost(
                            naive_bytes, num_hospitals, fed_res.n_rounds,
                        ),
                    # Eval-set size, needed to read an AUROC's precision.
                    "n_eval_windows": _ns.get(hid),
                    "n_eval_pos": _n_pos.get(hid),
                    # Whether this hospital passed the evidence floor and entered
                    # MACRO. Rows with False are kept for the record but should not
                    # be compared.
                    "in_macro": hid in macro_hids,
                    # 95 % CI half-width for every method. A difference of 0.0003
                    # means nothing next to +/-0.03.
                    "auroc_ci95_halfwidth_population_only": auroc_ci95_halfwidth(
                        pop_res.aucs.get(hid), _ns.get(hid), _n_pos.get(hid),
                    ),
                    "auroc_ci95_halfwidth_local_ft": auroc_ci95_halfwidth(
                        ft_res.aucs.get(hid), _ns.get(hid), _n_pos.get(hid),
                    ),
                    "auroc_ci95_halfwidth_fedavg": auroc_ci95_halfwidth(
                        fed_res.aucs.get(hid), _ns.get(hid), _n_pos.get(hid),
                    ),
                    "auroc_ci95_halfwidth_top_down": auroc_ci95_halfwidth(
                        td_res.aucs.get(hid), _ns.get(hid), _n_pos.get(hid),
                    ),
                    "auroc_ci95_halfwidth_hier": auroc_ci95_halfwidth(
                        hier_res.aucs.get(hid), _ns.get(hid), _n_pos.get(hid),
                    ),
                    # Rounds actually run
                    "rounds_local_ft": ft_res.n_rounds,
                    "rounds_fedavg": fed_res.n_rounds,
                    "rounds_top_down": td_res.n_rounds,
                    "rounds_hierarchical": hier_res.n_rounds,
                    # Round restored by model selection. 0 means the un-adapted
                    # model won, so that method's numbers equal population_only's in
                    # this cell. Methods agreeing because they all chose round 0 is
                    # not a finding.
                    "selected_round_local_ft": ft_res.selected_round,
                    "selected_round_fedavg": fed_res.selected_round,
                    "selected_round_top_down": td_res.selected_round,
                    "selected_round_hierarchical": hier_res.selected_round,
                    # How far the selected model moved from round 0. Non-zero with
                    # an unchanged AUROC means the eval set cannot resolve the
                    # change.
                    "selected_delta_norm_local_ft": ft_res.selected_delta_norm,
                    "selected_delta_norm_fedavg": fed_res.selected_delta_norm,
                    "selected_delta_norm_top_down": td_res.selected_delta_norm,
                    "selected_delta_norm_hierarchical": hier_res.selected_delta_norm,
                    # Patient adapter accept/reject counts (axis 2). Rejected
                    # patients fall back to the cohort model, so Delta_patient alone
                    # cannot show whether personalisation worked.
                    "patient_adapters_accepted": hier_res.extra.get(
                        "patient_accepted", {}).get(hid),
                    "patient_adapters_considered": hier_res.extra.get(
                        "patient_considered", {}).get(hid),
                    # Adapters that could not be judged (single-class validation
                    # slice). accepted=0 with unevaluable=considered means the layer
                    # was never tested, not that it failed.
                    "patient_adapters_unevaluable": hier_res.extra.get(
                        "patient_unevaluable", {}).get(hid),
                    # local_finetune's patient level, the control for the
                    # hierarchical one, paired against its own starting model.
                    "auc_local_finetune_patient": ft_patient_aucs.get(hid),
                    "patient_cohort_ref_auroc_local_ft": ft_patient_ref.get(hid),
                    "delta_patient_local_ft": (
                        None if (ft_patient_aucs.get(hid) is None
                                 or ft_patient_ref.get(hid) is None)
                        else float(ft_patient_aucs[hid]) - float(ft_patient_ref[hid])),
                    "patient_adapters_accepted_local_ft": ft_res.extra.get(
                        "patient_accepted", {}).get(hid),
                    "patient_adapters_considered_local_ft": ft_res.extra.get(
                        "patient_considered", {}).get(hid),
                    "patient_adapters_unevaluable_local_ft": ft_res.extra.get(
                        "patient_unevaluable", {}).get(hid),
                    "patient_pooled_pos": hier_res.extra.get(
                        "patient_pooled_pos", {}).get(hid),
                    # Wall-clock seconds (axis 6)
                    "wall_clock_local_ft": ft_res.wall_clock_sec,
                    "wall_clock_fedavg": fed_res.wall_clock_sec,
                    "wall_clock_top_down": td_res.wall_clock_sec,
                    "wall_clock_hierarchical": hier_res.wall_clock_sec,
                })

            # MACRO row, averaged only over hospitals that passed the evidence
            # floor, so it means the same thing at every alpha and for every task.
            rows.append({
                "task": task_name, "alpha": alpha, "hospital": "MACRO",
                # Configuration columns, so every row records the variant that
                # produced it.
                "cfg_train_head_bias_only": train_head_bias_only,
                "cfg_train_head": train_head,
                "cfg_use_flower": use_flower,
                "cfg_per_hospital_selection": per_hospital_selection,
                "cfg_split": split_label,
                "cfg_splits_dir": _sd or None,
                # The seeds actually used (after defaults), not only the ones passed
                # in.
                "cfg_seed": seed,
                "cfg_run_seed": run_seed,
                "cfg_rounds": getattr(cfg, "rounds", None),
                "cfg_cohort_local_steps": cohort_local_steps,
                "cfg_val_probe_frac": getattr(cfg, "val_probe_frac", None),
                "cfg_val_target_pos": getattr(cfg, "val_target_pos", None),
                "cfg_patient_weighting": patient_weighting_scheme,
                "cfg_cohort_weighting": cohort_weighting_scheme,
                "auc_population_only": _macro_over(pop_res.aucs, macro_hids),
                "auc_local_finetune": _macro_over(ft_res.aucs, macro_hids),
                "auc_standard_fedavg": _macro_over(fed_res.aucs, macro_hids),
                "auc_top_down_only": _macro_over(td_res.aucs, macro_hids),
                "auc_hierarchical_cohort": _macro_over(hier_res.aucs, macro_hids),
                "auc_hierarchical_patient": _macro_over(hier_patient_aucs, macro_hids),
                "auc_local_finetune_patient": _macro_over(ft_patient_aucs, macro_hids),
                "delta_patient_local_ft": _macro_over(
                    {h: (None if (ft_patient_aucs.get(h) is None
                                  or ft_patient_ref.get(h) is None)
                         else float(ft_patient_aucs[h]) - float(ft_patient_ref[h]))
                     for h in macro_hids}, macro_hids),
                "patient_adapters_accepted_local_ft": _sum_or_none(
                    ft_res.extra.get("patient_accepted", {}), macro_hids),
                "patient_adapters_considered_local_ft": _sum_or_none(
                    ft_res.extra.get("patient_considered", {}), macro_hids),
                "auprc_population_only": _macro_over(pop_res.auprcs, macro_hids),
                "auprc_local_finetune": _macro_over(ft_res.auprcs, macro_hids),
                "auprc_standard_fedavg": _macro_over(fed_res.auprcs, macro_hids),
                "auprc_top_down_only": _macro_over(td_res.auprcs, macro_hids),
                "auprc_hierarchical_cohort": _macro_over(hier_res.auprcs, macro_hids),
                "ece_population_only": _macro_over(pop_res.eces, macro_hids),
                "ece_local_finetune": _macro_over(ft_res.eces, macro_hids),
                "ece_standard_fedavg": _macro_over(fed_res.eces, macro_hids),
                "ece_top_down_only": _macro_over(td_res.eces, macro_hids),
                "ece_hierarchical_cohort": _macro_over(hier_res.eces, macro_hids),
                "pop_baseline": pop_baseline,
                # Size and membership of the MACRO average itself.
                "n_eval_windows": sum(
                    _ns.get(h) or 0 for h in macro_hids) or None,
                "n_eval_pos": sum(
                    _n_pos.get(h) or 0 for h in macro_hids) or None,
                # Approximate 95 % CI for the MACRO row: Hanley-McNeil on the pooled
                # counts. The reported number is a macro average of per-centre
                # AUROCs, which is a different estimator, so read this as an order
                # of magnitude. The paired bootstrap (scripts/paired_bootstrap.py)
                # is the proper test.
                "auroc_ci95_halfwidth_macro_approx": auroc_ci95_halfwidth(
                    _macro_over(pop_res.aucs, macro_hids),
                    sum(_ns.get(h) or 0 for h in macro_hids) or None,
                    sum(_n_pos.get(h) or 0 for h in macro_hids) or None),
                "macro_hospitals": " ".join(str(h) for h in macro_hids),
                "macro_excluded_hospitals": " ".join(str(h) for h in excluded_hids),
                "min_eval_pos": min_eval_pos,
                "min_eval_neg": min_eval_neg,
                # Validation scores. val_auroc_population_only is round 0 on
                # validation, i.e. the frozen model on the same surface as the
                # adapted methods. These cannot show whether adaptation helps,
                # because round 0 is a candidate and every method's validation score
                # is at least that high by construction. Compare configurations on
                # the probe columns and report only from test.
                "val_auroc_population_only": (
                    hier_res.val_auroc_round0 if hier_res.val_auroc_round0 is not None
                    else ft_res.val_auroc_round0),
                # Probe scores: the surface for comparing configurations without
                # reading test.
                "probe_auroc_population_only": _probe_pop_auroc,
                "probe_auroc_local_ft": ft_res.probe_auroc,
                "probe_auroc_fedavg": fed_res.probe_auroc,
                "probe_auroc_top_down": td_res.probe_auroc,
                "probe_auroc_hierarchical": hier_res.probe_auroc,
                # Every round's probe score, and how much the best one beats round
                # 0. If the gain is not positive, no round beat the population
                # model.
                "probe_round_aucs_hierarchical": " ".join(
                    "" if a is None else f"{a:.4f}"
                    for a in hier_res.extra.get("probe_round_aucs", [])) or None,
                "probe_best_round_hierarchical": _probe_best_round,
                "probe_best_gain_vs_round0": _probe_best_gain,
                "val_auroc_local_ft": ft_res.val_auroc,
                "val_auroc_fedavg": fed_res.val_auroc,
                "val_auroc_top_down": td_res.val_auroc,
                "val_auroc_hierarchical": hier_res.val_auroc,
                # Per-round validation trajectory per method. When a method selects
                # round 0, this shows whether later rounds were just below baseline
                # or falling apart.
                "val_traj_local_ft": " ".join(
                    "" if v is None else f"{v:.4f}" for _, v in ft_res.val_history) or None,
                "val_traj_fedavg": " ".join(
                    "" if v is None else f"{v:.4f}" for _, v in fed_res.val_history) or None,
                "val_traj_top_down": " ".join(
                    "" if v is None else f"{v:.4f}" for _, v in td_res.val_history) or None,
                "val_traj_hierarchical": " ".join(
                    "" if v is None else f"{v:.4f}" for _, v in hier_res.val_history) or None,
                "val_positives_per_hospital": (
                    None if not _val_pos_counts else
                    " ".join(str(v) for v in _val_pos_counts)),
                "selected_round_local_ft": ft_res.selected_round,
                "selected_round_fedavg": fed_res.selected_round,
                "selected_round_top_down": td_res.selected_round,
                "selected_round_hierarchical": hier_res.selected_round,
                # With per-centre selection, selected_round_* is None (centres chose
                # different rounds); these columns hold the rounds actually used.
                "selected_round_per_hosp_local_ft": _fmt_per_hosp_round(ft_res),
                "selected_round_per_hosp_top_down": _fmt_per_hosp_round(td_res),
                "selected_round_per_hosp_hierarchical": _fmt_per_hosp_round(hier_res),
                "selected_delta_norm_local_ft": ft_res.selected_delta_norm,
                "selected_delta_norm_fedavg": fed_res.selected_delta_norm,
                "selected_delta_norm_top_down": td_res.selected_delta_norm,
                "selected_delta_norm_hierarchical": hier_res.selected_delta_norm,
                # Keep 0 as 0: zero accepted adapters is a result, and None would
                # look like "the patient layer did not run".
                "patient_adapters_accepted": _sum_or_none(
                    hier_res.extra.get("patient_accepted", {}), macro_hids),
                "patient_adapters_considered": _sum_or_none(
                    hier_res.extra.get("patient_considered", {}), macro_hids),
                "patient_adapters_unevaluable": _sum_or_none(
                    hier_res.extra.get("patient_unevaluable", {}), macro_hids),
                "patient_pooled_pos": _sum_or_none(
                    hier_res.extra.get("patient_pooled_pos", {}), macro_hids),
                # Axes 2 (personalisation), 3 (forgetting) and 4 (upward), averaged
                # over hospitals.
                "fwt_hier_global": _macro_over(
                    hier_res.extra.get("fwt_global", {}), macro_hids),
                "delta_cohort_hier": _macro_over(_axis_dc, macro_hids),
                "delta_patient_hier": _macro_over(_axis_dp, macro_hids),
                "bwt_hier_cohort": _macro_over(_axis_bwt, macro_hids),
                "patient_propagation_gain": _macro_over(_axis_pg, macro_hids),
                "variance_hier_cohort": variance(
                    [hier_res.aucs.get(h) for h in macro_hids]),
                "variance_hier_patient": variance(
                    [hier_patient_aucs.get(h) for h in macro_hids]),
                # Axis 4, the upward half: population gain from cohort aggregation.
                "population_baseline_auc": pop_baseline,
                "population_gain_from_cohorts": population_gain_from_cohorts,
                # Axis 3: population stability across rounds
                "population_round_aucs": " ".join(
                    "" if a is None else f"{a:.4f}" for a in _pop_traj) or None,
                "population_stability_range": population_stability_range,
                # Axis 6: parameters updated per level
                "params_updated_cohort_per_round": _params_cohort,
                "params_updated_patient_per_round": _params_patient,
                "params_updated_population_per_sync": _params_population,
                "params_naive_full_model": _params_naive,
                "comm_bytes_cohort_per_round": cohort_bytes,
                "comm_bytes_patient_per_round": patient_bytes,
                "wall_clock_local_ft": ft_res.wall_clock_sec,
                "wall_clock_fedavg": fed_res.wall_clock_sec,
                "wall_clock_top_down": td_res.wall_clock_sec,
                "wall_clock_hierarchical": hier_res.wall_clock_sec,
                # Axis 6 also asks for convergence speed: rounds to the selected
                # model (selected_round_*) and the cost of one round at each level.
                "wall_clock_per_round_local_ft": (
                    ft_res.wall_clock_sec / ft_res.n_rounds if ft_res.n_rounds else None),
                "wall_clock_per_round_fedavg": (
                    fed_res.wall_clock_sec / fed_res.n_rounds if fed_res.n_rounds else None),
                "wall_clock_per_round_top_down": (
                    td_res.wall_clock_sec / td_res.n_rounds if td_res.n_rounds else None),
                "wall_clock_per_round_hierarchical": (
                    hier_res.wall_clock_sec / hier_res.n_rounds if hier_res.n_rounds else None),
                # Validation gain per round up to the selected model.
                "val_gain_per_round_hierarchical": (
                    (hier_res.val_auroc - hier_res.val_auroc_round0) / hier_res.selected_round
                    if (hier_res.selected_round and hier_res.val_auroc is not None
                        and hier_res.val_auroc_round0 is not None) else None),
            })
            pd.DataFrame(rows).to_csv(output_dir / "wp5_results.csv", index=False)

            # Save per-window predictions so method differences can be tested as
            # paired quantities later (scripts/paired_bootstrap.py). A finished cell
            # cannot rebuild them. About 1 MB per cell.
            _pred_blob = {}
            for _mname, _mres in (("population_only", pop_res),
                                  ("local_finetune", ft_res),
                                  ("standard_fedavg", fed_res),
                                  ("top_down_only", td_res),
                                  ("hierarchical", hier_res)):
                _mp = _mres.extra.get("test_preds", {}) or {}
                _ml = _mres.extra.get("test_labels", {}) or {}
                for _h, _pv in _mp.items():
                    if _pv is None:
                        continue
                    _pred_blob[f"{_mname}|{_h}|pred"] = np.asarray(_pv, dtype=np.float32)
                    if _ml.get(_h) is not None:
                        _pred_blob[f"{_mname}|{_h}|label"] = np.asarray(_ml[_h], dtype=np.int8)
            if _pred_blob:
                np.savez_compressed(
                    output_dir / f"predictions_{task_name}_a{alpha}.npz", **_pred_blob)
                print(f"  predictions saved: {len(_pred_blob)//2} (method,hospital) "
                      f"arrays -> predictions_{task_name}_a{alpha}.npz")
            # Axis 2: one row per patient, to study which patients benefit from
            # personalisation. Appended across cells so one file covers the whole
            # run.
            _prec = []
            for _meth, _res in (("hierarchical", hier_res), ("local_finetune", ft_res)):
                for _r in (_res.extra.get("patient_records", []) or []):
                    _r = dict(_r)
                    _r.setdefault("method", _meth)
                    _r.setdefault("task", task_name)
                    _r.setdefault("alpha", alpha)
                    _prec.append(_r)
            if _prec:
                patient_rows.extend(_prec)
                pd.DataFrame(patient_rows).to_csv(
                    output_dir / "patient_records.csv", index=False)
                _na = sum(1 for r in _prec if r["accepted"])
                _ne = sum(1 for r in _prec if r["evaluable"])
                print(f"  patient_records.csv: +{len(_prec)} rows "
                      f"({_ne} evaluable, {_na} accepted)")

    print("\n=== WP5 final summary ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved {output_dir / 'wp5_results.csv'}")


if __name__ == "__main__":
    main()
