"""Delta_patient must be a paired difference on the same evaluation windows.

The TD defines Delta_patient = Perf_patient - Perf_cohort. Patient models can only
be scored on train-partition patients (the centre must have their data), while the
cohort model is scored on the test partition. Subtracting those mixes a difference
between patient groups into the gain. So the un-adapted starting model is scored on
the same pooled patient windows, and the difference is taken against that.
"""
from __future__ import annotations

import pytest


def paired_delta(patient_auc, cohort_on_same_windows):
    if patient_auc is None or cohort_on_same_windows is None:
        return None
    return float(patient_auc) - float(cohort_on_same_windows)


def unpaired_delta(patient_auc, cohort_on_test_partition):
    """The old definition, kept only to show the difference."""
    if patient_auc is None or cohort_on_test_partition is None:
        return None
    return float(patient_auc) - float(cohort_on_test_partition)


def test_paired_delta_is_zero_when_no_adapter_is_accepted():
    """If every adapter is rejected, patients fall back to the cohort model, so both pooled scores use identical predictions and must cancel."""
    assert paired_delta(0.7421, 0.7421) == 0.0


def test_unpaired_delta_reports_a_spurious_gain_in_that_same_case():
    """The old definition reported +0.093 on class-aware mortality with 0 of 23 adapters accepted: purely a train/test population difference."""
    patient = cohort_same_windows = 0.7421
    cohort_on_test = 0.6943
    assert paired_delta(patient, cohort_same_windows) == 0.0
    assert unpaired_delta(patient, cohort_on_test) == pytest.approx(0.0478)


def test_paired_delta_is_negative_when_adapters_hurt():
    """When accepted adapters lose on held-out data, the paired difference shows a loss."""
    assert paired_delta(0.5536, 0.6100) < 0


def test_paired_delta_is_positive_only_when_personalisation_actually_helps():
    assert paired_delta(0.72, 0.70) == pytest.approx(0.02)


def test_none_propagates_rather_than_being_treated_as_zero():
    assert paired_delta(None, 0.7) is None
    assert paired_delta(0.7, None) is None


def test_harness_uses_the_paired_reference_for_the_reported_column():
    import inspect

    from cocofl.evaluation import run_evaluation as r

    src = inspect.getsource(r)
    assert '"patient_cohort_ref"' in src, "the paired reference must be computed"
    assert '"delta_patient_hier": _axis_dp[hid]' in src, (
        "the reported Δ_patient must use the paired axis value, not "
        "delta_patient(rec), which differences against the test partition"
    )
    assert '"delta_patient_hier_unpaired"' in src, (
        "keep the old definition under a name that says what it is, so nothing "
        "is silently redefined"
    )
