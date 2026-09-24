"""Patient-adapter acceptance must tell "rejected" apart from "could not be judged".

Acceptance compares AUROCs on the patient's own validation slice. On a single-class
slice AUROC is None and the adapter is rejected without being tested. For a terminal
outcome that is every case. On the real cohort at ``min_samples=60``::

    hospital_mortality   0 of 40 validation slices hold both classes
    readmission         37 of 40 validation slices hold both classes

So "0 of 23 accepted" for mortality means "never tested", not "failed". These tests
check that distinction, and the evidence floor that stops the pooled patient AUROC
from reporting 1.0000 off a single positive.
"""
from __future__ import annotations

from cocofl.personalisation.evaluate import auc_from_preds


# The root cause, as the harness sees it

def test_auc_is_none_on_a_single_class_slice():
    """This None is what turns into an automatic reject."""
    assert auc_from_preds([0.9, 0.2, 0.5], [0, 0, 0]).auc is None
    assert auc_from_preds([0.9, 0.2, 0.5], [1, 1, 1]).auc is None


def test_auc_is_defined_once_both_classes_appear():
    assert auc_from_preds([0.9, 0.1], [1, 0]).auc == 1.0


def _decide(a_pers, a_base):
    """Same accept/reject rule as the harness, including the evaluability split."""
    evaluable = a_pers is not None
    use_patient = evaluable and (a_base is None or a_pers >= a_base)
    return use_patient, (not evaluable)


def test_unevaluable_adapter_is_rejected_but_counted_separately():
    use, uneval = _decide(None, 0.7)
    assert use is False, "never adopt an adapter that was not evaluated"
    assert uneval is True, "and do not call it a rejection on merit"


def test_adapter_that_loses_on_merit_is_not_counted_as_unevaluable():
    use, uneval = _decide(0.60, 0.80)
    assert use is False and uneval is False


def test_adapter_that_wins_is_accepted():
    use, uneval = _decide(0.85, 0.80)
    assert use is True and uneval is False


def test_tie_goes_to_the_patient_adapter():
    use, _ = _decide(0.80, 0.80)
    assert use is True


def test_adapter_is_accepted_when_the_cohort_baseline_is_undefined():
    """If only the cohort side cannot be judged, the patient side still counts."""
    use, uneval = _decide(0.75, None)
    assert use is True and uneval is False


# The pooled patient AUROC needs an evidence floor

def _pooled_auc(preds, labels, floor):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos < floor or n_neg < floor:
        return None
    return auc_from_preds(preds, labels).auc


def test_single_positive_cannot_produce_a_perfect_score():
    """One positive must not give a perfect score (this produced a spurious Delta_patient of +0.093 on class-aware mortality)."""
    preds = [0.99] + [0.1] * 40
    labels = [1] + [0] * 40
    assert auc_from_preds(preds, labels).auc == 1.0, "unguarded, it is perfect"
    assert _pooled_auc(preds, labels, floor=10) is None, "guarded, it is withheld"


def test_floor_admits_a_pooled_set_with_real_evidence():
    preds = [0.9] * 12 + [0.1] * 30
    labels = [1] * 12 + [0] * 30
    assert _pooled_auc(preds, labels, floor=10) == 1.0


def test_floor_also_requires_enough_negatives():
    preds = [0.9] * 20 + [0.1] * 3
    labels = [1] * 20 + [0] * 3
    assert _pooled_auc(preds, labels, floor=10) is None


def test_floor_of_zero_restores_unguarded_behaviour():
    preds, labels = [0.99] + [0.1] * 5, [1] + [0] * 5
    assert _pooled_auc(preds, labels, floor=0) == 1.0
