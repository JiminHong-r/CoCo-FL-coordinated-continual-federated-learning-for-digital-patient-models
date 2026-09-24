"""local_finetune and top_down_only must be two different methods.

The TD lists them separately:

* local fine-tuning: per-centre and per-patient fine-tuning without coordination *
top-down only: hierarchical adaptation without upward propagation

What separates them is the downward flow. A locally fine-tuned patient starts from
the population model; a top-down patient starts from its centre's cohort model. If
both started from the cohort (and local_finetune just set ``eta_cohort=0``), they
would be the same method. The comparison then measures one thing: whether inheriting
the centre's knowledge is worth anything.
"""
from __future__ import annotations

import inspect

from cocofl.evaluation import baselines
from cocofl.hierarchical.coordinator import CoordinatorConfig


def test_patient_init_source_is_configurable():
    assert CoordinatorConfig().patient_init_from_population is False, (
        "hierarchical default: patients inherit their centre's model"
    )
    assert CoordinatorConfig(patient_init_from_population=True
                             ).patient_init_from_population is True


def test_local_finetune_disables_coordination_in_all_three_ways():
    src = inspect.getsource(baselines.run_local_finetune)
    assert "isolated_cfg.eta_cohort = 0.0" in src, "no patient->cohort aggregation"
    assert "isolated_cfg.eta_population = 0.0" in src, "no cohort->population"
    assert "population_update_freq = max(" in src, "no periodic sync"


def test_local_finetune_starts_patients_from_the_population():
    src = inspect.getsource(baselines.run_local_finetune)
    assert "isolated_cfg.patient_init_from_population = True" in src, (
        "without this it is the same method as top_down_only"
    )


def test_top_down_keeps_patients_on_their_cohort():
    """top_down_only must not set the flag: starting patients from their centre is what makes it hierarchical."""
    src = inspect.getsource(baselines.run_top_down_only)
    assert "patient_init_from_population" not in src


def test_local_finetune_can_take_real_patient_streams():
    sig = inspect.signature(baselines.run_local_finetune)
    assert "patient_streams" in sig.parameters
    assert sig.parameters["patient_streams"].default is None, (
        "off by default: enabling changes an existing baseline's numbers"
    )


def test_patient_layer_evaluation_is_shared_not_duplicated():
    """Both methods use one shared accept/reject implementation."""
    assert callable(baselines.evaluate_patient_layer)
    sig = inspect.signature(baselines.evaluate_patient_layer)
    assert "init_states" in sig.parameters, (
        "the shared helper must be parameterised by what the patient starts from"
    )
