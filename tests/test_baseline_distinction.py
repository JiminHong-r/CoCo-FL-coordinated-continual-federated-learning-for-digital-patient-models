"""``local_finetune`` and ``top_down_only`` must be two different methods.

The TD lists them as separate baselines:

  • Local fine-tuning: per-centre **and per-patient** fine-tuning without coordination
  • Top-down only: hierarchical adaptation without upward propagation

As implemented, ``local_finetune`` passed EMPTY patient streams, so it was a
one-level baseline and the hierarchical patient layer had no control to be
compared against. Worse, the obvious repair does not work: patients always
initialise from their centre's cohort state, so simply handing ``local_finetune``
real streams and setting ``eta_cohort=0`` makes it *identical* to
``top_down_only`` — same initialisation, same training, no aggregation either
way. Two names, one method.

The distinction that makes both meaningful is the DOWNWARD flow: a locally
fine-tuned patient starts from the population model (no hierarchy), a top-down
patient starts from its centre's cohort model (hierarchy used downward, upward
disabled). The contrast then measures exactly one thing — whether inheriting
centre-level knowledge is worth anything.
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
    """It must NOT set the flag — inheriting the centre is what makes it
    'hierarchical adaptation'."""
    src = inspect.getsource(baselines.run_top_down_only)
    assert "patient_init_from_population" not in src


def test_local_finetune_can_take_real_patient_streams():
    sig = inspect.signature(baselines.run_local_finetune)
    assert "patient_streams" in sig.parameters
    assert sig.parameters["patient_streams"].default is None, (
        "off by default: enabling changes an existing baseline's numbers"
    )


def test_patient_layer_evaluation_is_shared_not_duplicated():
    """One accept/reject implementation for both methods — a second copy is how
    the levels drifted apart before."""
    assert callable(baselines.evaluate_patient_layer)
    sig = inspect.signature(baselines.evaluate_patient_layer)
    assert "init_states" in sig.parameters, (
        "the shared helper must be parameterised by what the patient starts from"
    )
