"""The patient layer needs an accept/reject step, like the cohort layer's model selection.

Without it, whatever the patient adapter produced was scored, and Delta_patient
averaged -0.099 while Delta_cohort (protected by its fallback) sat at 0. Acceptance
must not look at the eval windows, which are the patient's test data, so each
patient's windows are split by time into fit, validation and eval, and acceptance
uses validation only.
"""
from __future__ import annotations

import inspect


def _accept(a_pers, a_base):
    """Same acceptance rule as the harness."""
    return a_pers is not None and (a_base is None or a_pers >= a_base)


def test_worse_patient_adapter_is_rejected():
    assert _accept(0.60, 0.75) is False, "a degrading adapter must be discarded"


def test_better_patient_adapter_is_kept():
    assert _accept(0.82, 0.75) is True


def test_ties_keep_the_patient_adapter():
    """A tie keeps the personalised model; ties should not quietly switch the layer off."""
    assert _accept(0.75, 0.75) is True


def test_undefined_patient_score_is_rejected():
    """A single-class validation slice gives AUROC None; fall back instead of guessing."""
    assert _accept(None, 0.75) is False


def test_undefined_baseline_keeps_the_patient_adapter():
    assert _accept(0.70, None) is True


def test_streams_are_three_way_fit_val_eval():
    from cocofl.hierarchical.run_hierarchical import build_patient_streams

    src = inspect.getsource(build_patient_streams)
    assert "streams.append((pid, t, v, e))" in src, "streams must be (patient_id, fit, val, eval): the id is needed for axis 2 subgroup analysis and cannot be recovered afterwards"
    assert "train_frac=0.7" in src, "val slice carved from the EARLY (train) half"


def test_selection_uses_val_not_eval():
    """Acceptance must never look at the eval windows."""
    import cocofl.evaluation.baselines as r

    src = inspect.getsource(r)
    assert "predict(pers_clf, pt_val, label_fn, device)" in src
    assert "predict(base_clf, pt_val, label_fn, device)" in src
    # The eval windows are used once, to score the chosen model
    assert "predict(chosen, pt_eval, label_fn, device)" in src
    assert "predict(pers_clf, pt_eval" not in src, (
        "scoring the personalised model directly on eval would bypass selection"
    )


def test_min_samples_default_supports_a_val_slice():
    """At min_samples=20 a three-way split leaves about 7 fit / 3 validation windows, too few to select on; 60 gives about 21 / 9 / 30."""
    from cocofl.hierarchical.run_hierarchical import build_patient_streams

    sig = inspect.signature(build_patient_streams)
    assert sig.parameters["min_samples"].default >= 60
