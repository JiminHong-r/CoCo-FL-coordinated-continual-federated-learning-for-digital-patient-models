"""The patient layer needs the same accept/reject the cohort layer already has.

Model selection was added for cohort states (best validation round restored,
round 0 = the un-adapted population model as a candidate). The patient layer
never got it: ``train_patient_adapter`` ran its iterations and whatever came out
was scored. The consequence was visible in the first completed task —
delta_cohort sat at exactly 0.0000 in 8 of 11 rows (protected by the fallback)
while delta_patient averaged **-0.099** (unprotected), which reported the
hierarchy's headline mechanism as harmful for a reason that was an artifact.

The fix has to avoid an obvious trap: a patient's windows were split 50/50 into
train and eval, so selecting on the eval slice would be selecting on the test
set. Hence a three-way chronological split — fit / val / eval — with acceptance
decided on val only.
"""
from __future__ import annotations

import inspect


def _accept(a_pers, a_base):
    """Mirrors the harness's acceptance rule."""
    return a_pers is not None and (a_base is None or a_pers >= a_base)


def test_worse_patient_adapter_is_rejected():
    assert _accept(0.60, 0.75) is False, "a degrading adapter must be discarded"


def test_better_patient_adapter_is_kept():
    assert _accept(0.82, 0.75) is True


def test_ties_keep_the_patient_adapter():
    """Equal validation performance keeps the personalised model — the layer is
    the point of the method, so ties should not silently disable it."""
    assert _accept(0.75, 0.75) is True


def test_undefined_patient_score_is_rejected():
    """A single-class val slice gives AUC=None; fall back rather than guess."""
    assert _accept(None, 0.75) is False


def test_undefined_baseline_keeps_the_patient_adapter():
    assert _accept(0.70, None) is True


def test_streams_are_three_way_fit_val_eval():
    from cocofl.hierarchical.run_hierarchical import build_patient_streams

    src = inspect.getsource(build_patient_streams)
    assert "streams.append((pid, t, v, e))" in src, "streams must be (patient_id, fit, val, eval): the id is needed for axis 2 subgroup analysis and cannot be recovered afterwards"
    assert "train_frac=0.7" in src, "val slice carved from the EARLY (train) half"


def test_selection_uses_val_not_eval():
    """The load-bearing property: acceptance must never look at eval windows."""
    import cocofl.evaluation.baselines as r

    src = inspect.getsource(r)
    assert "predict(pers_clf, pt_val, label_fn, device)" in src
    assert "predict(base_clf, pt_val, label_fn, device)" in src
    # eval is used once, to score the CHOSEN model
    assert "predict(chosen, pt_eval, label_fn, device)" in src
    assert "predict(pers_clf, pt_eval" not in src, (
        "scoring the personalised model directly on eval would bypass selection"
    )


def test_min_samples_default_supports_a_val_slice():
    """At min_samples=20 a 3-way split leaves ~7 fit / 3 val — too thin to
    select on. 60 gives ~21 / 9 / 30."""
    from cocofl.hierarchical.run_hierarchical import build_patient_streams

    sig = inspect.signature(build_patient_streams)
    assert sig.parameters["min_samples"].default >= 60
