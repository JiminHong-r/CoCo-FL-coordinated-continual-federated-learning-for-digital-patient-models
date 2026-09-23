"""Validation sets must be sized by evidence, not by window count.

Model selection ranks rounds by validation AUROC. At hospital_mortality's 2.09 %
prevalence a 2,000-window validation set contains ~42 positives, which puts the
AUROC standard error near ±0.07 — so round-to-round gaps of ~0.01 are noise and
"best round" is close to arbitrary. That single fact is consistent with every
selection anomaly observed: round 0 winning in some cells, round 3 being chosen
on validation and then losing 0.016 on test, fedavg always landing on the final
round. readmission at 20.6 % gets ~410 positives from the same 2,000 windows and
its selection behaves sensibly, which is the control.
"""
from __future__ import annotations

import math


def sized(val_per_hosp: int, prevalence: float, target_pos: int) -> int:
    """Mirrors the harness: never shrink below the configured floor."""
    if target_pos <= 0 or prevalence <= 0:
        return val_per_hosp
    return max(val_per_hosp, math.ceil(target_pos / prevalence))


def test_disabled_by_default_keeps_the_flat_size():
    assert sized(2000, 0.0209, target_pos=0) == 2000


def test_rare_outcome_gets_a_much_larger_val_set():
    n = sized(2000, 0.0209, target_pos=200)
    assert n > 9000, n
    assert n * 0.0209 >= 200


def test_common_outcome_needs_no_enlargement():
    """readmission already carries enough positives, so nothing changes."""
    assert sized(2000, 0.206, target_pos=200) == 2000


def test_never_shrinks_below_the_configured_floor():
    assert sized(5000, 0.5, target_pos=200) == 5000


def test_the_42_positive_problem_is_what_this_fixes():
    before = 2000 * 0.0209
    after = sized(2000, 0.0209, 200) * 0.0209
    assert before < 50, "the status quo: selection on ~42 positives"
    assert after >= 200, "after: selection on evidence"


def test_zero_prevalence_does_not_divide_by_zero():
    assert sized(2000, 0.0, target_pos=200) == 2000


# --- validation-space comparison: what it can and cannot answer -----------

def test_population_only_has_a_validation_score_at_all():
    """It is round 0 — the un-adapted model — measured on the val loaders. It was
    computed for selection and then discarded, so a method's validation number
    had nothing to be compared against."""
    from cocofl.evaluation.result import MethodResult
    r = MethodResult(val_auroc=0.83, val_auroc_round0=0.81)
    assert r.val_auroc_round0 == 0.81


def test_val_comparison_is_biased_by_construction():
    """Round 0 is a selection CANDIDATE, so the selected round's validation score
    is >= round 0's for every method, always. "Adaptation beats the population
    model on validation" is therefore true by construction and cannot be used to
    decide whether adaptation helps — which is why config choice needs a separate
    probe slice and the reported answer needs the test set."""
    def select(val_by_round):
        best = 0
        for r, v in enumerate(val_by_round):
            if v > val_by_round[best]:
                best = r
        return best, val_by_round[best]

    for traj in ([0.80, 0.70, 0.60], [0.80, 0.90, 0.85], [0.80, 0.80, 0.80]):
        _, best_val = select(traj)
        assert best_val >= traj[0], "selection can never land below round 0"


def test_a_validation_gain_need_not_transfer_to_test():
    """Observed on class-aware mortality: validation picked round 3, and test was
    0.016 WORSE than round 0. Selecting on a 42-positive val set does this."""
    val = {0: 0.8450, 3: 0.8478}
    test = {0: 0.8774, 3: 0.8610}
    assert val[3] > val[0], "validation preferred round 3"
    assert test[3] < test[0], "test disagreed"
