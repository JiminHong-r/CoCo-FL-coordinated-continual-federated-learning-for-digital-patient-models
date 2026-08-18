"""WP5 main harness: run all baselines + hierarchical across α sweep.

Per (task, α) it records AUC of each method:
- ``population_only``    : baseline (no adaptation)
- ``local_finetune``     : per-centre adapter without coordination
- ``top_down_only``      : hierarchical without upward propagation
- ``hierarchical``       : full bidirectional (our method)

For each method, AUC is reported PER hospital + macro-averaged.

The harness intentionally uses a SMALL real config so the full sweep is
feasible on a single GPU within ~1 h. Scale via ``+rounds`` / ``+n_patient_iters``
/ ``+cohort_local_steps`` / ``+n_pop_eval`` for paper-grade runs.

Usage::

    cocofl_wp5 \\
        model_path=$PROJECT/models/population/best_model.pt \\
        +train_dir=$PROJECT/data/tokenized/mimic_train \\
        input_dir=$PROJECT/data/tokenized/mimic_test \\
        output_dir=$PROJECT/wp5_artifacts \\
        +tasks=hospital_mortality \\
        +alphas=0.1,1.0,10.0 \\
        +num_hospitals=3 +rounds=2 \\
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
from torch.utils.data import DataLoader, Subset

from ..constants import STATIC_DATA_FN, horizon_microseconds
from ..data.static_features import build_static_arrays
from ..hierarchical.cohort import CohortConfig
from ..hierarchical.coordinator import CoordinatorConfig, HierarchicalCoordinator
from ..hierarchical.patient import PatientConfig
from ..hierarchical.run_hierarchical import (
    build_hospital_train_loader, build_patient_streams,
    build_population_classifier, evaluate_with_states,
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
    """First ``n`` indices, or all of them when ``n`` is 0/None (uncapped).

    ``idxs`` is already in shuffled patient order (the split JSON is built from
    a seeded shuffle), so a prefix is an unbiased subsample — the cap costs
    precision, not correctness. Every method is scored on the *same* prefix, so
    between-method comparisons stay paired.
    """
    return list(idxs) if not n else list(idxs)[:n]


def _build_pop_eval_loader(hospital_eval_loaders, test_ds, collate_fn, n_pop_eval,
                             batch_size=1):
    """Pooled held-out reference set for the frozen population model.

    Built from the hospitals' EVAL partitions, not their training ones. This
    feeds ``pop_baseline``, which Axis 3's ``fwt`` subtracts from the cohort
    AUROC — and the cohort AUROC is measured on the eval partition. Sourcing the
    baseline from training windows made that subtraction compare two numbers
    measured on different data, so any train/eval difficulty gap was reported as
    forward transfer.
    """
    pop_eval_indices = sorted({
        idx for ld in hospital_eval_loaders.values()
        for idx in ld.dataset.indices
    })[:n_pop_eval]
    return make_eval_loader(test_ds, pop_eval_indices, collate_fn, batch_size=batch_size)


def _eval_sizes(*results) -> tuple[dict, dict]:
    """First non-empty ``(ns, n_pos)`` among ``results``.

    Every method scores the same loaders, so eval-set size is a property of the
    (task, α) cell and not of any one method. Falling back across results is
    what lets the WP4 ablations — which run ``hierarchical`` alone, with no
    ``population_only`` — still report sizes and confidence intervals.
    """
    for r in results:
        if r.ns:
            return r.ns, r.n_pos
    return {}, {}


def _reportable_hospitals(hids, ns: dict, n_pos: dict,
                          min_pos: int, min_neg: int) -> list:
    """Hospitals whose eval set can support an AUROC at all.

    A hospital qualifies when it has at least ``min_pos`` positives AND
    ``min_neg`` negatives. Hospitals with unknown sizes are kept — absent
    evidence of a problem, excluding them would silently drop real cells.
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

    A plain ``sum(...) or None`` turns a real zero into a blank cell, and zero is
    exactly the value that matters for patient-adapter acceptance: "0 of 23
    accepted" is a finding, "blank" reads as "the layer never ran".
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
    """Run hierarchical method with early stopping. Returns a MethodResult
    with patient AUCs and per-round trajectory under ``extra``.

    ``eval_loaders`` is scored **once**, at the end, for the reported test
    metrics. The per-round BWT / propagation-gain trajectory is scored on
    ``traj_loaders`` instead (defaulting to ``eval_loaders``): it needs two
    extra passes *per round*, so pointing it at a capped loader is what keeps a
    full-size ``eval_loaders`` affordable. BWT and propagation gain are
    round-to-round *differences* on a fixed sample, so a capped trajectory set
    is the right trade — the level it reports is noisier, the difference is not.
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
    # Spec WP5 axis 3 ("population model stability: tracking performance on a
    # held-out global validation set across all update rounds") and axis 4
    # ("improvement in POPULATION model performance attributable to cohort-level
    # aggregation"). Both need the population model scored per round, which the
    # harness never did — it only ever scored the frozen population once. The
    # upward half of the bidirectional loop was therefore unmeasured, even though
    # it is the project's central claim.
    population_round_aucs: list[float | None] = []
    probe_round_aucs: list[float | None] = []
    _per_hosp_best = dict(_val0)
    _per_hosp_round = {h: 0 for h in _val0}
    _per_hosp_state = {h: {k: v.clone() for k, v in st.items()}
                       for h, st in coord.cohort_states.items()}
    for r in range(1, max_rounds + 1):
        log = coord.run_round()
        # Spec WP5 axis 4: AUC AFTER cohort-only training but BEFORE patient
        # aggregation lets us attribute the improvement to patient feedback.
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
        # Test-set AUC after this round (BWT trajectory). Skip hospitals with
        # no test-partition indices (rare under low α).
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
        # Attribution: improvement from patient feedback per hospital.
        propagation_gains.append({
            hid: (round_aucs[hid] - pre_patient_aucs[hid])
                  if (round_aucs[hid] is not None and pre_patient_aucs.get(hid) is not None)
                  else None
            for hid in round_aucs
        })
        # Per-round score on the PROBE slice. This is the achievability bound:
        # if no round beats round 0 here, then no amount of better model
        # selection can make adaptation win, because there is nothing to select.
        # Distinguishing "selection is too noisy to find the good round" from
        # "there is no good round" is otherwise impossible, and they call for
        # opposite conclusions — the first is a measurement fix, the second is a
        # genuine null. Probe is used because it never drives selection, so
        # reading it costs nothing and leaves the test set sealed.
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

        # Population-level trajectory (axes 3 + 4). The population adapter shares
        # the cohort adapter's shape and slot, so it loads the same way. Before
        # the first sync there is no adapter and the population model is exactly
        # the frozen baseline.
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

        # Val-set macro AUC for early stopping (separate from test for unbiased eval)
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
        # Per-centre bookkeeping — see baselines._round_loop_with_early_stopping
        # for why one globally-chosen round is the wrong default under a
        # heterogeneous split.
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

    # Restore the best-validation round. Without this the harness reported
    # whatever the final round produced — and with patience=999 the stopping
    # rule can never fire, so there was no model selection at all. Round 0 (the
    # identity adapter, i.e. the population model) is a candidate, which is what
    # makes "adaptation is never worse than its own starting point" hold.
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

    # Config-selection surface (never used to pick a round) — see
    # MethodResult.probe_auroc for why neither val nor test can do this job.
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

    # Spec WP5 axis 3: FORWARD transfer = "generalisation to new patients or
    # tasks". Each cohort model is scored on the GLOBAL held-out set — patients
    # belonging to no centre — against the frozen population model on the same
    # set. The previous fwt() was ``cohort_auc - pop_baseline`` on the centre's
    # OWN test partition, which is Δ_cohort rescaled, not forward transfer: it
    # never left the centre the model was adapted to.
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

    # Full metrics on test set (AUROC + AUPRC + ECE)
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

    # Patient-level AUCs: pool predictions across up to ``patient_eval_n``
    # patient streams per hospital. Pooling addresses the previous-run issue
    # where any single stream's late samples were often single-class and
    # produced AUC=None — pooling across multiple patients restores class
    # diversity and gives a meaningful AUC.
    # Patient level via the SHARED implementation, so `hierarchical` and the
    # local fine-tuning baseline are judged by identical accept/reject machinery.
    # This was a second inline copy of that logic; two copies of an accept/reject
    # rule is exactly how the levels drifted apart before.
    from .baselines import evaluate_patient_layer

    # How many patients per hospital are pooled into the patient-level AUROC.
    # The legacy default derives it from ``n_iters``, which are unrelated knobs:
    # at n_iters=10 it silently resolves to 2 patients, and with min_samples=4
    # each contributes ~2 predictions — so Δ_patient rested on ~4 predictions
    # per hospital. Set ``patient_eval_patients`` to control it explicitly.
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

    # Knobs
    num_hospitals = int(getattr(cfg, "num_hospitals", 3))
    # Accept alphas as a Hydra ListConfig (preferred: ``+alphas=[0.1,1.0,10.0]``),
    # a Python list, or a comma-separated string.
    alphas_raw = getattr(cfg, "alphas", "0.1,1.0,10.0")
    try:
        # ListConfig / list iteration
        alphas = [float(a) for a in alphas_raw]
    except TypeError:
        alphas = [float(a) for a in str(alphas_raw).split(",")]
    # Early stopping: replaces fixed ``rounds`` with ``max_rounds`` + ``patience``.
    max_rounds = int(getattr(cfg, "max_rounds", getattr(cfg, "rounds", 15)))
    patience = int(getattr(cfg, "patience", 3))
    min_rounds = int(getattr(cfg, "min_rounds", 2))
    patients_per_round = int(getattr(cfg, "patients_per_round", 2))
    max_patients_per_hospital = int(getattr(cfg, "max_patients_per_hospital", 3))
    n_patient_iters = int(getattr(cfg, "n_patient_iters", 5))
    cohort_bottleneck = int(getattr(cfg, "cohort_bottleneck", 64))
    patient_bottleneck = cohort_bottleneck  # must match for upward prop
    cohort_local_steps = int(getattr(cfg, "cohort_local_steps", 5))
    # WP4 tasks 1-2: cohort fine-tunes at a MEDIUM lr; patient adapters adapt
    # FAST at a HIGHER lr. Keep configurable but default patient_lr > cohort_lr.
    cohort_lr = float(getattr(cfg, "cohort_lr", 1e-3))
    patient_lr = float(getattr(cfg, "patient_lr", 5e-3))
    n_pop_eval = int(getattr(cfg, "n_pop_eval", 50))
    # 0 = uncapped (score the hospital's whole test/val partition). The default
    # stays capped so existing configs reproduce byte-for-byte.
    eval_per_hosp = int(getattr(cfg, "eval_per_hospital", 200))
    val_per_hosp = int(getattr(cfg, "val_per_hospital", eval_per_hosp))
    # Per-round trajectory (BWT, propagation gain) — two extra passes per round,
    # so keep this capped even when the reported test set is not.
    traj_per_hosp = int(getattr(cfg, "traj_eval_per_hospital", eval_per_hosp))
    # Size the VALIDATION set by how many positives it will contain, not by a
    # flat window count. Model selection ranks rounds by validation AUROC, and at
    # hospital_mortality's 2.09 % prevalence a 2,000-window val set holds ~42
    # positives — an AUROC standard error around ±0.07, so round-to-round
    # differences of ~0.01 are indistinguishable from noise and "best round" is
    # close to a coin toss. That is consistent with everything observed: round 0
    # winning in some cells, round 3 being selected on validation and then losing
    # 0.016 on test, fedavg picking the final round. readmission at 20.6 %
    # prevalence already gets ~410 positives from the same 2,000 windows, and its
    # selection behaves sensibly — which is the control for this explanation.
    #
    # 0 disables (flat val_per_hospital, the previous behaviour).
    val_target_pos = int(getattr(cfg, "val_target_pos", 0))
    # Fraction of each centre's VAL partition held back as a config-selection
    # probe, never used to pick a round. 0 = disabled (all of val drives
    # selection, the previous behaviour). Only needed while searching over
    # configurations; the final reported run does not require it.
    val_probe_frac = float(getattr(cfg, "val_probe_frac", 0.0))
    # Minimum evidence for a hospital cell to enter the MACRO average. A
    # Dirichlet draw at low α leaves centres with a handful of patients — at
    # α=0.1 hospital 2 holds 69 — and the AUROCs they produce are coin flips
    # (readmission α=0.1 scored 0.4 / 0.5 / 0.3 / 0.8 across four methods on one
    # such centre). Averaged in with equal weight, one 10-pair centre moves MACRO
    # as much as a 100,000-pair one, which is what makes MACRO incomparable
    # across α and across tasks. Cells below the floor are still written to the
    # per-hospital rows — nothing is hidden — but flagged and excluded from
    # MACRO. Set both to 0 to reproduce the old unguarded averages.
    min_eval_pos = int(getattr(cfg, "min_eval_pos", 10))
    min_eval_neg = int(getattr(cfg, "min_eval_neg", 10))
    # Length-bucketed eval batching (see evaluation/batching.py). Predictions
    # are identical to batch_size=1; this only stops the T-LSTM from running one
    # sequence at a time.
    eval_batch_size = int(getattr(cfg, "eval_batch_size", 1))
    # Cohort-training batch size. Unlike eval batching this DOES change the
    # optimisation — ``cohort_local_steps`` counts optimiser steps, so at batch B
    # a centre sees ``steps * B`` windows per round, and each step is an average
    # over B examples rather than a single one. It defaults to 1 so existing
    # configs reproduce exactly; raise it only when the intent is to train a
    # centre on a meaningful fraction of its partition, and say so in the writeup.
    cohort_batch_size = int(getattr(cfg, "cohort_batch_size", 1))
    # Whether FedAvg trains under the same CL regulariser as every other method.
    # True keeps the comparison about aggregation topology alone; False gives the
    # vanilla FedAvg baseline (what the harness did implicitly, by omission).
    # TD baseline: "Standard FedAvg: classical federated averaging WITHOUT
    # continual learning or hierarchy". So this defaults to FALSE.
    #
    # It defaulted to True, which made the baseline FedAvg + knowledge
    # distillation — a stronger, non-classical competitor, and not the reference
    # point the TD defines. The reasoning behind True was to hold CL constant so
    # "hierarchical vs FedAvg" isolated topology; that is a legitimate ABLATION,
    # but it is not the TD's baseline, and running it as the baseline made the
    # hierarchical method look worse against a benchmark that had been handed the
    # hierarchy's regulariser. Set fedavg_use_cl=true to get the CL-matched
    # variant explicitly, and report it as an ablation rather than the baseline.
    fedavg_use_cl = bool(getattr(cfg, "fedavg_use_cl", False))
    # Class-stratified sampling for cohort training, matching how the WP3
    # population head was fitted (all positives kept, negatives capped). Without
    # it the head was fitted on a ~20 %-positive draw while adaptation trained on
    # the natural ~2 %, so at a truncated step budget the adapter saw ~24x fewer
    # positives per unit of compute — a difference in TRAINING PROCEDURE, not in
    # method. pos_weight is recomputed from the resampled draw so the correction
    # is not applied twice.
    # Patient-training batch size. The patient adapter has the same 66 K
    # parameters as the cohort adapter but was training at batch 1: with
    # n_patient_iters=10 that is TEN windows, so it memorised them and lost on
    # validation — which is why patient selection rejected every adapter.
    patient_batch_size = int(getattr(cfg, "patient_batch_size", 8))
    cohort_balanced_sampling = bool(getattr(cfg, "cohort_balanced_sampling", False))
    cohort_max_neg_per_pos = float(getattr(cfg, "cohort_max_neg_per_pos", 3.0))
    # Patient-layer sample size (Δ_patient). ``patient_min_samples`` sets the
    # windows a patient must have to qualify; half become eval windows.
    patient_min_samples = int(getattr(cfg, "patient_min_samples", 4))
    patient_eval_patients = getattr(cfg, "patient_eval_patients", None)
    if patient_eval_patients is not None:
        patient_eval_patients = int(patient_eval_patients)
    eta_cohort = float(getattr(cfg, "eta_cohort", 0.3))
    eta_population = float(getattr(cfg, "eta_population", 0.1))
    seed = int(getattr(cfg, "seed", 42))
    # TWO seeds, deliberately separate.
    #
    # ``seed`` fixes the DATA: the Dirichlet partition, the balanced-sampling
    # draw and the dataloader order. ``run_seed`` fixes the TRAINING: torch's
    # global RNG, which is what initialises each adapter's down-projection.
    #
    # They must be separable. Torch's global RNG was never seeded here at all, so
    # no cell was reproducible — re-running a config gave a different adapter
    # init, and with a validation surface flat enough that round selection is
    # decided by noise, that alone moved the selected round (a same-config
    # replicate picked round 3 vs round 0). Seeding only ``seed`` would fix that
    # but make replication impossible, because ``seed`` also redraws the
    # partition: varying it would confound training noise with a different split
    # of patients into centres, which are different questions.
    #
    # So: hold ``seed`` fixed across a campaign for one partition; vary
    # ``run_seed`` to measure run-to-run variance on that partition.
    run_seed = int(getattr(cfg, "run_seed", seed))
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    print(f"  seeds: data seed={seed}, run seed={run_seed}")
    tasks_filter = getattr(cfg, "tasks", None)
    # Fixed-splits: when ``splits_dir`` is set the runner loads pre-computed
    # patient partitions (train / val / test per hospital, per α) from JSON
    # files; otherwise it falls back to an inline Dirichlet draw.
    splits_dir = getattr(cfg, "splits_dir", None)

    Vocabulary.from_path(test_dir)
    with open(test_dir / STATIC_DATA_FN, "rb") as f:
        test_static = pickle.load(f)
    test_cat, test_num = build_static_arrays(test_static)
    # NOTE: the real collate is built per-task INSIDE the task loop, once the
    # dataset exists — it needs the dataset to reconstruct each window's real
    # inter-event times (WP3 task 1). This placeholder exists only for code paths
    # that run before a task is selected.
    collate_fn = make_collate(test_cat, test_num)

    # The population classifier is per-TASK: its head is task-specific, so it is
    # rebuilt inside the task loop below. Without ``population_head_dir`` the head
    # is randomly initialised and population_only scores at chance — see
    # build_population_classifier.
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

    # Catastrophic-forgetting mitigation knobs (Spec WP4 task 4). Set strategy
    # to "distill" to add a KD term against the population (cohort) and cohort
    # (patient) teachers.
    cohort_cl_strategy = str(getattr(cfg, "cohort_cl_strategy", "none"))
    cohort_cl_alpha = float(getattr(cfg, "cohort_cl_alpha", 0.3))
    cohort_cl_temperature = float(getattr(cfg, "cohort_cl_temperature", 2.0))
    patient_cl_strategy = str(getattr(cfg, "patient_cl_strategy", "none"))
    patient_cl_alpha = float(getattr(cfg, "patient_cl_alpha", 0.3))
    patient_cl_temperature = float(getattr(cfg, "patient_cl_temperature", 2.0))

    # --- Spec WP4 task 3: upward-propagation weighting ------------------------
    # How patient deltas are weighted when they are aggregated into the cohort
    # (and cohort states into the population). The TD asks for schemes based on
    # data volume, uncertainty and recency; all four are implemented in
    # ``upward_prop._weights_for`` but this harness never read the knob, so every
    # reported run silently used "volume".
    # --- Method filter --------------------------------------------------------
    # Which of the five methods to actually run. Defaults to all, which is what
    # the headline WP5 table needs. The WP4 task-3/5 ablations only vary knobs
    # that affect upward propagation, so `population_only`, `local_finetune` and
    # `standard_fedavg` are bit-identical across their arms — running them once
    # per arm would triple the cost for no information. A skipped method leaves
    # its CSV columns empty rather than being silently reported as something.
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
    # Per-level schemes (Spec WP4 task 3 defines two upward mechanisms). Unset
    # means "use weighting_scheme at that level", which reproduces the old
    # single-knob behaviour exactly.
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
    # --- two method changes, both OFF by default -----------------------------
    # Each centre keeps its own best round rather than one globally-chosen round.
    per_hospital_selection = bool(getattr(cfg, "per_hospital_selection", False))
    # Unfreeze only the classifier head's intercepts, so a per-centre label shift
    # can be corrected where it should be instead of through the adapter.
    train_head_bias_only = bool(getattr(cfg, "train_head_bias_only", False))
    # Train the FULL classifier head per centre, not just its intercepts.
    #
    # Motivated by a measurement, not a guess: the cross-centre transfer test
    # (scripts/cross_centre_transfer.py) shows a per-centre LOGISTIC head — about
    # 1,000 parameters on frozen features — beats a pooled head fitted on the same
    # budget by +0.004 to +0.016 AUROC, while a 66,112-parameter cohort adapter
    # feeding a FROZEN head recovers essentially none of it. That gap says the
    # limitation is where the capacity sits, not how much there is. Letting the
    # head move per centre tests exactly that.
    #
    # Off by default: it changes what a "cohort model" is, so any cell run with it
    # is not comparable to the campaign's other cells.
    train_head = bool(getattr(cfg, "train_head", False))
    # TD baselines list "local fine-tuning: per-centre AND per-patient fine-tuning
    # without coordination". The harness passed empty patient streams, making it a
    # one-level baseline with nothing for the hierarchical patient layer to be
    # compared against. Off by default: enabling it changes that baseline's
    # numbers and takes it from ~1.2 h to ~3 h per cell.
    # TD baseline: "Local fine-tuning: per-centre AND per-patient fine-tuning
    # without coordination". Default TRUE so the reported baseline is the one the
    # TD defines; it roughly triples that method's cost. Set false only to
    # reproduce the older per-centre-only runs.
    local_finetune_patients = bool(getattr(cfg, "local_finetune_patients", True))
    # Spec WP4 task 5: "after N patient updates" — a distinct trigger from round
    # frequency, since one round contains many patient updates.
    population_update_after_n_patients = getattr(
        cfg, "population_update_after_n_patients", None)
    if population_update_after_n_patients is not None:
        population_update_after_n_patients = int(population_update_after_n_patients)

    # --- Spec WP4 task 5: coordination protocol -------------------------------
    # Population sync fires on FREQUENCY (every N rounds) OR on DIVERGENCE (L2
    # between a cohort state and the last broadcast population state exceeds the
    # threshold). ``divergence_threshold=None`` disables the divergence arm,
    # which is what every reported run used — so only the frequency arm has ever
    # been exercised. ``population_update_freq`` defaults to max_rounds, i.e. one
    # sync on the final round; set it lower to study the protocol.
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
        # Top-down refresh strength: blend cohort toward population instead of
        # overwriting (refresh_blend=1.0 = legacy overwrite). Default 0.3.
        refresh_blend=float(getattr(cfg, "refresh_blend", 0.3)),
        patient_rotation=bool(getattr(cfg, "patient_rotation", False)),
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
        # Rebuild the collate with the dataset so the T-LSTM receives REAL Δt.
        # Without this every window was fed elapsed=1.0, which pins the time decay
        # at the constant 0.7615 and runs the backbone outside the distribution it
        # was trained on (see cocofl.data.elapsed).
        collate_fn = make_collate(test_cat, test_num, dataset=test_ds)
        label_fn = make_label_fn(task_name)
        population_clf = build_population_classifier(
            model_path, device,
            head_path=(Path(population_head_dir) / f"{task_name}.pt")
            if population_head_dir else None,
        )
        # Class weight for cohort/patient training. The population head is fit
        # with a weighted loss on a stratified draw (WP3); fine-tuning it here
        # with an UNWEIGHTED loss on the centre's natural prevalence drags it
        # back toward the base rate, which is why adaptation was scoring BELOW
        # the frozen population model. One weight per task is enough: WP2's
        # characterisation shows positive rates are uniform across centres
        # (mortality 2.02-2.16 % at every alpha).
        from ..downstream.probe import stratified_subsample, window_labels
        _y_task = window_labels(
            test_ds, POSITIVE_TOKENS[task_name], horizon_microseconds(task_name),
        )
        _prev = float(_y_task.mean())
        task_pos_weight = ((1.0 - _prev) / _prev) if _prev > 0 else None
        print(f"  task prevalence {_prev:.3%} -> cohort/patient pos_weight="
              f"{task_pos_weight:.1f}" if task_pos_weight else "  no positives")
        # Per-task validation size, so every task's selection rests on comparable
        # evidence rather than on comparable window counts.
        val_per_hosp_task = val_per_hosp
        if val_target_pos > 0 and _prev > 0:
            needed = int(-(-val_target_pos // _prev))  # ceil
            val_per_hosp_task = max(val_per_hosp, needed)
            print(f"  val sizing: target {val_target_pos} positives at "
                  f"{_prev:.3%} prevalence -> val_per_hospital="
                  f"{val_per_hosp_task} (was {val_per_hosp})")
        # Applied to every method: all five train through these configs, so the
        # weighting is symmetric and the comparison stays like-for-like.
        coord_cfg_template.cohort_cfg.pos_weight = task_pos_weight
        coord_cfg_template.patient_cfg.pos_weight = task_pos_weight

        for alpha in alphas:
            print(f"\n=== task={task_name}  α={alpha} ===")
            hospital_val_loaders: dict = {}
            if splits_dir is not None:
                # Use pre-computed fixed split (Spec WP2 task 4: no leakage, reproducible)
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
                # Split each centre's val partition into a SELECTION half and a
                # config PROBE half. The probe never drives round selection, so
                # it is a clean surface for comparing configurations while the
                # test set stays untouched for the final report.
                _val_sel_idx, _val_probe_idx = {}, {}
                for hid, idxs in val_split.indices_per_hospital.items():
                    if not idxs:
                        continue
                    capped = _cap(idxs, val_per_hosp_task)
                    if val_probe_frac > 0.0:
                        n_probe = int(len(capped) * val_probe_frac)
                        # Interleave rather than slice: the indices are time-ordered,
                        # so a contiguous tail would make the probe a strictly LATER
                        # period than the selection half and confound the comparison.
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
                # Per-round trajectory set: a prefix of the SAME test indices, so
                # BWT stays a like-for-like difference across rounds.
                hospital_traj_loaders = {
                    hid: make_eval_loader(test_ds, _cap(idxs, traj_per_hosp), collate_fn,
                                           batch_size=eval_batch_size)
                    for hid, idxs in test_split.indices_per_hospital.items() if idxs
                }
                # Patient streams come from TRAIN partition — those patients have their
                # local model adapted; their deltas can propagate up to the cohort.
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
                    # Hold out a DISJOINT eval slice. The eval loaders here used
                    # to be built from the train loader's own indices, so the
                    # eval set *was* the train set and every method was scored on
                    # data it had just trained on. No shipped script hits this
                    # branch (they all pass splits_dir), but nothing stopped a
                    # bare `cocofl_evaluate` call from doing so silently.
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
                # Without fixed splits there is no separate val partition; fall back
                # to using the test loaders for early stopping. (Less rigorous; only
                # recommended for smoke tests.)
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

            # A skipped method yields an empty MethodResult, so every downstream
            # ``.aucs.get(hid)`` returns None and its CSV columns stay blank.
            pop_res = ft_res = fed_res = td_res = hier_res = MethodResult()
            pop_baseline = None
            # Held-out global validation set for the population level (axes 3+4).
            # Built for every cell, not just when population_only runs: the WP4
            # ablations run `hierarchical` alone and still owe the population
            # stability trajectory and the cohort->population gain.
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
                    # TD baseline: "per-centre AND per-patient fine-tuning". Off
                    # unless requested, because switching it on changes an
                    # existing baseline's numbers and roughly triples its cost.
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
                # Per-level comm cost (Spec WP5 axis 6)
                cohort_bytes = communication_bytes(fed_res.extra["server_state"])
                # Patient adapter same shape as cohort (bottleneck must match)
                patient_bytes = cohort_bytes
                # Naïve full-model FL comparison: every centre sends the entire
                # backbone state every round (Spec WP5 axis 6 reference baseline).
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
            # How much evidence model selection actually had. At 2 % prevalence a
            # 2,000-window val set holds ~42 positives, which is why "best round"
            # was close to arbitrary; this column makes that visible per cell
            # instead of leaving it to be inferred.
            # Frozen population model on the probe slice — the reference every
            # config comparison is made against.
            _probe_pop_auroc = None
            if hospital_probe_loaders:
                _pp = {hid: eval_auc(population_clf, ld, label_fn, device).auc
                       for hid, ld in hospital_probe_loaders.items()}
                _probe_pop_auroc = macro_average(_pp.values())
                print(f"  probe: population_only={_probe_pop_auroc}")

            # ACHIEVABILITY BOUND. Round 0 on the probe slice IS the frozen
            # population model, so comparing the best round against it answers the
            # question no single selected number can: is there a good round at all?
            #
            # This is an upper bound, NOT a reportable method — picking a round by
            # looking at the probe is still selection. Its value is diagnostic:
            #   gain > 0  -> a better round exists; selection is the bottleneck,
            #               and enlarging the val set is the right fix.
            #   gain <= 0 -> no round beats not adapting. The null is structural
            #               and no configuration search can overturn it.
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

            # --- Spec WP5 axes 3, 4, 6 -----------------------------------
            _pop_traj = hier_res.extra.get("population_round_aucs", []) or []
            _pop_traj_vals = [a for a in _pop_traj if a is not None]
            # Axis 4: improvement in the POPULATION model attributable to
            # cohort-level aggregation — the upward half of the loop, previously
            # unmeasured. Axis 3: the same trajectory is the population stability
            # curve the TD asks to be tracked across all update rounds.
            population_gain_from_cohorts = (
                (_pop_traj_vals[-1] - pop_baseline)
                if (_pop_traj_vals and pop_baseline is not None) else None
            )
            population_stability_range = (
                (max(_pop_traj_vals) - min(_pop_traj_vals))
                if len(_pop_traj_vals) >= 2 else None
            )
            # Axis 6 asks for parameters per level, not just bytes.
            _params_cohort = (hier_res.extra.get("cohort_state_params")
                              or parameters_updated(fed_res.extra.get("server_state")))
            # Patient adapters share the cohort adapter's shape — upward
            # propagation requires it (patient_bottleneck == cohort_bottleneck).
            _params_patient = _params_cohort
            _params_population = hier_res.extra.get("population_state_params")
            _params_naive = naive_full_model_parameters(population_clf)
            # local_finetune now HAS a patient level (the TD's baseline is
            # "per-centre AND per-patient"), and it costs real GPU time — it is why
            # that method went from ~1.2 h to ~3.7 h per cell. Its results were
            # computed and then discarded, so the control for the hierarchical
            # patient layer was being paid for and thrown away.
            ft_patient_aucs = ft_res.extra.get("patient_aucs", {})
            ft_patient_ref = ft_res.extra.get("patient_cohort_ref", {})
            hier_patient_aucs = hier_res.extra.get("patient_aucs", {})
            per_round_aucs = hier_res.extra.get("per_round_cohort_aucs", [])
            propagation_gains = hier_res.extra.get("propagation_gains", [])

            # Build per-hospital records
            from .metrics import bwt
            all_hids = set(hospital_train_loaders) | set(hospital_eval_loaders)
            # Eval-set sizes for this cell, from whichever method ran. Needed for
            # the CI half-widths and the MACRO evidence floor below.
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
            # Keyed by hospital, not appended in order, so the MACRO aggregate can
            # apply the same evidence floor the AUROC columns do.
            _axis_dc, _axis_dp, _axis_bwt, _axis_pg = {}, {}, {}, {}
            for hid in sorted(all_hids):
                rec = AucPerLevel(
                    task=task_name, hospital=hid,
                    population=pop_res.aucs.get(hid),
                    cohort=hier_res.aucs.get(hid),
                    patient=hier_patient_aucs.get(hid),
                )
                hospital_bwt = bwt([r.get(hid) for r in per_round_aucs])
                # Spec WP5 axis 4: per-hospital average AUC gain from patient
                # upward propagation (post-patient AUC minus pre-patient AUC,
                # averaged across rounds). Positive => patients help cohort.
                hospital_prop_gain = macro_average([
                    g.get(hid) for g in propagation_gains
                ])
                # Axes 2/3/4 are computed per hospital but were never carried up
                # to the MACRO row, so delta_cohort / delta_patient / BWT /
                # propagation gain read as NaN there — and those are required
                # WP5 deliverables. Collect them for the macro aggregate below.
                _axis_dc[hid] = delta_cohort(rec)
                # PAIRED: patient vs the un-adapted model on the SAME windows.
                # rec.cohort is the cohort model on the TEST partition, a different
                # population, so delta_patient(rec) was subtracting unlike things.
                _pc_ref = hier_res.extra.get("patient_cohort_ref", {}).get(hid)
                _p_auc = hier_patient_aucs.get(hid)
                _axis_dp[hid] = (None if (_p_auc is None or _pc_ref is None)
                                 else float(_p_auc) - float(_pc_ref))
                _axis_bwt[hid] = hospital_bwt
                _axis_pg[hid] = hospital_prop_gain
                rows.append({
                    "task": task_name, "alpha": alpha, "hospital": hid,
                    # AUROC for all methods
                    "auc_population_only": pop_res.aucs.get(hid),
                    "auc_local_finetune": ft_res.aucs.get(hid),
                    "auc_standard_fedavg": fed_res.aucs.get(hid),
                    "auc_top_down_only": td_res.aucs.get(hid),
                    "auc_hierarchical_cohort": hier_res.aucs.get(hid),
                    "auc_hierarchical_patient": hier_patient_aucs.get(hid),
                    # AUPRC for all methods (Spec axis 1)
                    "auprc_population_only": pop_res.auprcs.get(hid),
                    "auprc_local_finetune": ft_res.auprcs.get(hid),
                    "auprc_standard_fedavg": fed_res.auprcs.get(hid),
                    "auprc_top_down_only": td_res.auprcs.get(hid),
                    "auprc_hierarchical_cohort": hier_res.auprcs.get(hid),
                    # ECE for all methods (Spec axis 1)
                    "ece_population_only": pop_res.eces.get(hid),
                    "ece_local_finetune": ft_res.eces.get(hid),
                    "ece_standard_fedavg": fed_res.eces.get(hid),
                    "ece_top_down_only": td_res.eces.get(hid),
                    "ece_hierarchical_cohort": hier_res.eces.get(hid),
                    # Personalisation gain + transfer
                    "pop_baseline": pop_baseline,
                    "delta_cohort_hier": delta_cohort(rec),
                    "delta_patient_hier": _axis_dp[hid],
                    "patient_cohort_ref_auroc": hier_res.extra.get(
                        "patient_cohort_ref", {}).get(hid),
                    # Retained for continuity: the OLD unpaired definition.
                    "delta_patient_hier_unpaired": delta_patient(rec),
                    # Same-centre difference, i.e. Δ_cohort rescaled. Kept for
                    # continuity; it is NOT forward transfer.
                    "fwt_hier_samecentre": fwt(rec, pop_baseline),
                    # TD axis 3 FWT: generalisation OUTSIDE the centre.
                    "fwt_hier_global": hier_res.extra.get("fwt_global", {}).get(hid),
                    "bwt_hier_cohort": hospital_bwt,
                    # Spec WP5 axis 4: cohort improvement attributable to
                    # patient-level upward propagation (avg across rounds).
                    "patient_propagation_gain": hospital_prop_gain,
                    # Communication (Spec axis 6) — per-round adapter cost,
                    # plus naïve full-model FL baseline + total over training.
                    "comm_bytes_cohort_per_round": cohort_bytes,
                    "comm_bytes_patient_per_round": patient_bytes,
                    "comm_naive_full_model_bytes_per_round": naive_bytes,
                    "comm_savings_ratio_vs_naive": savings,
                    # Comm costs are derived from the FedAvg server state, so
                    # they are undefined when that method is not run (the WP4
                    # ablations). None here, rather than a fabricated number.
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
                    # Eval-set size + the precision it implies. Without these an
                    # AUROC in this table cannot be read honestly.
                    "n_eval_windows": _ns.get(hid),
                    "n_eval_pos": _n_pos.get(hid),
                    # Whether this cell cleared the evidence floor and therefore
                    # entered the MACRO row. False cells are kept for the record
                    # but must not be compared across methods.
                    "in_macro": hid in macro_hids,
                    # CI half-width for EVERY method, not just two. A method
                    # comparison is only readable when each side carries its own
                    # resolution: 0.8764 vs 0.8761 means nothing next to ±0.03.
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
                    # Rounds actually used (early stopping)
                    "rounds_local_ft": ft_res.n_rounds,
                    "rounds_fedavg": fed_res.n_rounds,
                    "rounds_top_down": td_res.n_rounds,
                    "rounds_hierarchical": hier_res.n_rounds,
                    # Which round model selection restored, per method. 0 means
                    # the un-adapted population model won, so that method's
                    # numbers ARE population_only's and every ablation knob was a
                    # no-op in this cell. Reading agreement between methods as a
                    # finding, when all of them selected round 0, is the trap
                    # these columns exist to close.
                    "selected_round_local_ft": ft_res.selected_round,
                    "selected_round_fedavg": fed_res.selected_round,
                    "selected_round_top_down": td_res.selected_round,
                    "selected_round_hierarchical": hier_res.selected_round,
                    # Distance the selected model moved from round 0. Non-zero
                    # with unchanged AUROC = the eval set cannot resolve it.
                    "selected_delta_norm_local_ft": ft_res.selected_delta_norm,
                    "selected_delta_norm_fedavg": fed_res.selected_delta_norm,
                    "selected_delta_norm_top_down": td_res.selected_delta_norm,
                    "selected_delta_norm_hierarchical": hier_res.selected_delta_norm,
                    # Patient-adapter accept/reject outcome (Axis 2). Δ_patient is
                    # floored by the fallback to the cohort model, so the delta
                    # alone cannot show whether personalisation worked.
                    "patient_adapters_accepted": hier_res.extra.get(
                        "patient_accepted", {}).get(hid),
                    "patient_adapters_considered": hier_res.extra.get(
                        "patient_considered", {}).get(hid),
                    # How many could not be judged at all (single-class val
                    # slice). accepted=0 with unevaluable=considered means the
                    # patient layer was never tested, NOT that it failed.
                    "patient_adapters_unevaluable": hier_res.extra.get(
                        "patient_unevaluable", {}).get(hid),
                    # The TD baseline's patient level — the control for the
                    # hierarchical one, and paired against its own starting model.
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
                    # Wall-clock seconds (Spec axis 6)
                    "wall_clock_local_ft": ft_res.wall_clock_sec,
                    "wall_clock_fedavg": fed_res.wall_clock_sec,
                    "wall_clock_top_down": td_res.wall_clock_sec,
                    "wall_clock_hierarchical": hier_res.wall_clock_sec,
                })

            # Macro-average row
            # MACRO is averaged ONLY over hospitals that cleared the evidence
            # floor, so it means the same thing at every α and for every task.
            rows.append({
                "task": task_name, "alpha": alpha, "hospital": "MACRO",
                # Provenance. Without these a results CSV does not record which
                # variant produced it, and two cells that differ only in a flag
                # are indistinguishable after the fact — which is how the
                # bias-only arm ended up confounded with per-centre selection
                # and unattributable. A row must describe its own configuration.
                "cfg_train_head_bias_only": train_head_bias_only,
                "cfg_train_head": train_head,
                "cfg_per_hospital_selection": per_hospital_selection,
                "cfg_split": getattr(cfg, "split_scheme", None),
                # Record the seeds that were USED, not the ones that were passed.
                # `getattr(cfg, "seed", None)` wrote a blank whenever the runner
                # omitted `+seed=`, even though the code had already defaulted to
                # 42 and the cell was perfectly reproducible — so the CSV
                # understated its own reproducibility. TD §3.10 requires seeds to
                # be "set and recorded"; recording the resolved values is what
                # makes the second half true.
                "cfg_seed": seed,
                "cfg_run_seed": run_seed,
                "cfg_rounds": getattr(cfg, "rounds", None),
                "cfg_cohort_local_steps": getattr(
                    getattr(cfg, "cohort", None), "local_steps", None),
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
                # Sizes and membership of the MACRO average itself, so a reader
                # can see how much evidence the headline number rests on and
                # which centres it dropped.
                "n_eval_windows": sum(
                    _ns.get(h) or 0 for h in macro_hids) or None,
                "n_eval_pos": sum(
                    _n_pos.get(h) or 0 for h in macro_hids) or None,
                # CI at the AGGREGATE level. The whole "these are ties" argument
                # rests on comparing differences against a confidence interval, and
                # the MACRO row — the row people read — had none: the five CI
                # columns are written per hospital only.
                #
                # APPROXIMATION, stated because it matters: this is Hanley-McNeil
                # on the POOLED counts, whereas the reported figure is a macro
                # average of per-centre AUROCs. The two are not the same estimator,
                # so treat this as the right order of magnitude for "is a
                # difference resolvable", not as an exact interval. A paired test
                # on shared windows is the correct instrument and is not yet built.
                "auroc_ci95_halfwidth_macro_approx": auroc_ci95_halfwidth(
                    _macro_over(pop_res.aucs, macro_hids),
                    sum(_ns.get(h) or 0 for h in macro_hids) or None,
                    sum(_n_pos.get(h) or 0 for h in macro_hids) or None),
                "macro_hospitals": " ".join(str(h) for h in macro_hids),
                "macro_excluded_hospitals": " ".join(str(h) for h in excluded_hids),
                "min_eval_pos": min_eval_pos,
                "min_eval_neg": min_eval_neg,
                # Validation-space scores. ``val_auroc_population_only`` is the
                # round-0 validation macro, i.e. the frozen population model
                # measured on the SAME surface the adapted methods are — without
                # it a method's val number had nothing to compare against.
                #
                # These do NOT answer "does adaptation help". Round 0 is a
                # selection candidate, so every method's val_auroc is >= the
                # round-0 value by construction. Use them to see how far
                # selection moved and on how much evidence; use the probe slice
                # (val_probe_frac) to choose between configurations, and the test
                # columns only for the final report.
                "val_auroc_population_only": (
                    hier_res.val_auroc_round0 if hier_res.val_auroc_round0 is not None
                    else ft_res.val_auroc_round0),
                # THE surface for choosing between configurations. population_only's
                # probe score is the frozen model on the same slice, so
                # "does config X let a method beat population_only?" is answerable
                # here without ever reading the test columns.
                "probe_auroc_population_only": _probe_pop_auroc,
                "probe_auroc_local_ft": ft_res.probe_auroc,
                "probe_auroc_fedavg": fed_res.probe_auroc,
                "probe_auroc_top_down": td_res.probe_auroc,
                "probe_auroc_hierarchical": hier_res.probe_auroc,
                # Every round scored on the probe slice, plus whether the BEST of
                # them beats round 0. If probe_best_beats_round0 is False the null
                # is structural: there is no round for selection to find, and the
                # result is not a tuning artefact.
                "probe_round_aucs_hierarchical": " ".join(
                    "" if a is None else f"{a:.4f}"
                    for a in hier_res.extra.get("probe_round_aucs", [])) or None,
                "probe_best_round_hierarchical": _probe_best_round,
                "probe_best_gain_vs_round0": _probe_best_gain,
                "val_auroc_local_ft": ft_res.val_auroc,
                "val_auroc_fedavg": fed_res.val_auroc,
                "val_auroc_top_down": td_res.val_auroc,
                "val_auroc_hierarchical": hier_res.val_auroc,
                # Per-round validation trajectory for every method. Computed for
                # selection and then discarded, which left the most important
                # follow-up question unanswerable: when a method selects round 0,
                # were the later rounds a whisker below baseline (the method is
                # close, and something small could tip it) or collapsing (something
                # is actively wrong)? Those call for opposite responses and the
                # single selected number cannot distinguish them.
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
                # Under per-centre selection `selected_round_*` is None by
                # construction (centres chose different rounds); these carry the
                # rounds actually used, so a row is never self-contradictory.
                "selected_round_per_hosp_local_ft": _fmt_per_hosp_round(ft_res),
                "selected_round_per_hosp_top_down": _fmt_per_hosp_round(td_res),
                "selected_round_per_hosp_hierarchical": _fmt_per_hosp_round(hier_res),
                "selected_delta_norm_local_ft": ft_res.selected_delta_norm,
                "selected_delta_norm_fedavg": fed_res.selected_delta_norm,
                "selected_delta_norm_top_down": td_res.selected_delta_norm,
                "selected_delta_norm_hierarchical": hier_res.selected_delta_norm,
                # `or None` would be wrong here: ZERO accepted adapters is the
                # single most informative value this column can take, and
                # collapsing it to None makes "none were accepted" look like
                # "the patient layer did not run". Keep 0 as 0; use None only
                # when no hospital reported a count at all.
                "patient_adapters_accepted": _sum_or_none(
                    hier_res.extra.get("patient_accepted", {}), macro_hids),
                "patient_adapters_considered": _sum_or_none(
                    hier_res.extra.get("patient_considered", {}), macro_hids),
                "patient_adapters_unevaluable": _sum_or_none(
                    hier_res.extra.get("patient_unevaluable", {}), macro_hids),
                "patient_pooled_pos": _sum_or_none(
                    hier_res.extra.get("patient_pooled_pos", {}), macro_hids),
                # Axis 2 (personalisation gain), Axis 3 (forgetting),
                # Axis 4 (upward propagation) — macro over hospitals.
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
                # --- Spec WP5 axis 4: the UPWARD half of the loop --------
                # "improvement in population model performance attributable to
                # cohort-level aggregation". Previously absent: only the
                # patient->cohort direction was measured.
                "population_baseline_auc": pop_baseline,
                "population_gain_from_cohorts": population_gain_from_cohorts,
                # --- axis 3: population stability across all update rounds ---
                "population_round_aucs": " ".join(
                    "" if a is None else f"{a:.4f}" for a in _pop_traj) or None,
                "population_stability_range": population_stability_range,
                # --- axis 6: parameters per level, the TD's unit -------------
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
                # Axis 6 asks for wall-clock AND "convergence speed across
                # adaptation levels". Convergence speed has two components, both
                # already known per method: how many rounds were needed to reach
                # the selected model (selected_round_*, i.e. rounds-to-best) and
                # how long a round costs at each level.
                "wall_clock_per_round_local_ft": (
                    ft_res.wall_clock_sec / ft_res.n_rounds if ft_res.n_rounds else None),
                "wall_clock_per_round_fedavg": (
                    fed_res.wall_clock_sec / fed_res.n_rounds if fed_res.n_rounds else None),
                "wall_clock_per_round_top_down": (
                    td_res.wall_clock_sec / td_res.n_rounds if td_res.n_rounds else None),
                "wall_clock_per_round_hierarchical": (
                    hier_res.wall_clock_sec / hier_res.n_rounds if hier_res.n_rounds else None),
                # Validation gain per round to the selected model — the rate at
                # which adaptation actually bought anything.
                "val_gain_per_round_hierarchical": (
                    (hier_res.val_auroc - hier_res.val_auroc_round0) / hier_res.selected_round
                    if (hier_res.selected_round and hier_res.val_auroc is not None
                        and hier_res.val_auroc_round0 is not None) else None),
            })
            pd.DataFrame(rows).to_csv(output_dir / "wp5_results.csv", index=False)

            # Per-window predictions, so method differences can be tested as the
            # PAIRED quantities they are. Every method scores identical windows, so
            # the interval on a DIFFERENCE is far tighter than the independent
            # per-method intervals (±0.016 here) suggest — but a finished cell
            # cannot regenerate predictions, so they must be persisted now or the
            # comparison can never be tested properly.
            # ~36 K windows x 5 methods x float32 is under 1 MB per cell.
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
            # Spec WP5 axis 2: per-patient rows, for "when and for which patient
            # subgroups personalisation is most beneficial". Appended across
            # (task, α) cells so one file covers the whole run.
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
