"""Adaptation must not be trained with a loss that is degenerate at these prevalences.

WP3 fits the population head with a class-weighted loss on a stratified draw
(~20 % positives). WP4/WP5 then fine-tune that head — but did so with an
UNWEIGHTED BCE on a centre's natural ~2 % prevalence, where the loss is
minimised by drifting toward the base rate. The observable consequence was that
every adapted method scored BELOW the frozen population model, and FedAvg —
which averages three separately drifted heads — scored worst of all. That reads
as "adaptation doesn't work" when it is really "the training objective is wrong".

``pos_weight`` is applied at both cohort and patient level, and to every method,
so the comparison stays symmetric.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from cocofl.hierarchical.cohort import CohortConfig
from cocofl.hierarchical.patient import PatientConfig


def _prevalence_to_pos_weight(p):
    return (1.0 - p) / p


def test_configs_expose_pos_weight_defaulting_to_unweighted():
    assert CohortConfig().pos_weight is None
    assert PatientConfig().pos_weight is None
    assert CohortConfig(pos_weight=46.9).pos_weight == 46.9
    assert PatientConfig(pos_weight=46.9).pos_weight == 46.9


def test_pos_weight_matches_measured_prevalence():
    """mortality runs at ~2.09 %, so the positive class needs ~47x the weight."""
    pw = _prevalence_to_pos_weight(0.0209)
    assert 46 < pw < 48


def test_weighted_loss_penalises_missing_the_rare_class():
    """The whole point: at 2 % prevalence, an all-negative predictor must not win."""
    y = torch.zeros(100)
    y[:2] = 1.0                      # 2 % positives
    always_negative = torch.full((100,), -5.0)

    unweighted = F.binary_cross_entropy_with_logits(always_negative, y)
    weighted = F.binary_cross_entropy_with_logits(
        always_negative, y, pos_weight=torch.tensor(_prevalence_to_pos_weight(0.02)),
    )
    assert weighted > unweighted * 5, (
        "an unweighted loss barely penalises ignoring the positive class, which "
        "is what let cohort training degrade a balanced head"
    )


def test_weighted_loss_prefers_a_discriminating_predictor():
    y = torch.zeros(100)
    y[:2] = 1.0
    pw = torch.tensor(_prevalence_to_pos_weight(0.02))

    always_negative = torch.full((100,), -5.0)
    discriminating = torch.full((100,), -5.0)
    discriminating[:2] = 5.0

    assert (F.binary_cross_entropy_with_logits(discriminating, y, pos_weight=pw)
            < F.binary_cross_entropy_with_logits(always_negative, y, pos_weight=pw))


def test_cohort_and_patient_training_use_the_weight():
    import inspect

    from cocofl.hierarchical import cohort, patient

    assert "pos_weight=_pw" in inspect.getsource(cohort), (
        "cohort training must apply pos_weight to its task BCE"
    )
    assert "pos_weight=_pw" in inspect.getsource(patient), (
        "patient training must apply pos_weight to its task BCE"
    )


def test_harness_sets_the_weight_for_every_method():
    import inspect

    import cocofl.evaluation.run_evaluation as r

    src = inspect.getsource(r)
    assert "coord_cfg_template.cohort_cfg.pos_weight = task_pos_weight" in src
    assert "coord_cfg_template.patient_cfg.pos_weight = task_pos_weight" in src


def test_fedavg_gets_the_same_regulariser_as_every_other_method():
    """FedAvg used to ignore cl_strategy structurally — it was called without a
    teacher, so 'hierarchical vs FedAvg' compared topology+distillation against
    topology alone. That confound moved the result in BOTH directions depending
    on whether the population head was random or trained."""
    import inspect

    from cocofl.evaluation.baselines import run_standard_fedavg

    src = inspect.getsource(run_standard_fedavg)
    assert "teacher=teacher" in src, "FedAvg must receive the distillation teacher"
    assert "replay_buffer=replay_buffer" in src, "FedAvg must receive the replay buffer"
    assert "use_cl" in inspect.signature(run_standard_fedavg).parameters, (
        "a vanilla-FedAvg escape hatch must remain available"
    )


def test_fedavg_cl_defaults_on_so_the_comparison_is_symmetric():
    import inspect

    from cocofl.evaluation.baselines import run_standard_fedavg

    assert inspect.signature(run_standard_fedavg).parameters["use_cl"].default is True


def test_balanced_cohort_sampling_matches_the_population_head_procedure():
    """The population head was fitted on a class-stratified draw; adaptation
    sampled naturally. Same imbalance handling must apply at both levels or the
    comparison is between training procedures, not methods."""
    import numpy as np

    from cocofl.downstream.probe import stratified_subsample

    # A centre at the measured mortality prevalence.
    y = np.zeros(100_000, dtype=np.int8)
    y[:2_090] = 1                                   # 2.09 %
    rng = np.random.default_rng(0)

    sel = stratified_subsample(y, range(len(y)), 0, rng, max_neg_per_pos=3.0)
    kept = y[np.asarray(sel)]

    assert kept.sum() == 2_090, "every positive must be kept"
    assert 0.20 < kept.mean() < 0.30, "negatives capped at 3:1 -> ~25 % positive"
    assert len(sel) < len(y), "negatives must actually be subsampled"


def test_balanced_sampling_recomputes_pos_weight():
    """Applying pos_weight from the NATURAL rate on top of a balanced draw would
    double-correct; the harness must derive it from the resampled draw."""
    import inspect

    import cocofl.evaluation.run_evaluation as r

    src = inspect.getsource(r)
    assert "cohort_balanced_sampling" in src
    # pos_weight is re-derived inside the balanced branch, not reused.
    assert src.count("coord_cfg_template.cohort_cfg.pos_weight") >= 2, (
        "pos_weight must be recomputed after resampling"
    )
