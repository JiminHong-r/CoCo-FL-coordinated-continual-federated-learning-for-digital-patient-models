"""Validation sets can be sized by number of positives, not by window count.

Round selection ranks rounds by validation AUROC. At mortality's 2.09 % prevalence
2,000 windows hold about 42 positives, so the AUROC standard error is near +/-0.07
and gaps of 0.01 between rounds are noise. Readmission at 20.6 % gets about 410
positives from the same 2,000 windows, and its selection behaves sensibly.
"""
from __future__ import annotations

import math


def sized(val_per_hosp: int, prevalence: float, target_pos: int) -> int:
    """Same rule as the harness: never below the configured size."""
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
    """Readmission already has enough positives, so nothing changes."""
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


# What validation-space comparisons can and cannot answer

def test_population_only_has_a_validation_score_at_all():
    """population_only's validation score is round 0 (the un-adapted model) on the validation loaders, recorded so methods have something to compare with."""
    from cocofl.evaluation.result import MethodResult
    r = MethodResult(val_auroc=0.83, val_auroc_round0=0.81)
    assert r.val_auroc_round0 == 0.81


def test_val_comparison_is_biased_by_construction():
    """The selected round's validation score is always at least round 0's, because round 0 is a candidate.

    So "adaptation beats the population model on validation" is always true and says
    nothing; comparing configurations needs the probe slice, and the reported answer
    needs the test set.
    """
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
    """A validation gain need not carry over to test: on class-aware mortality validation picked round 3, which was 0.016 worse than round 0 on test."""
    val = {0: 0.8450, 3: 0.8478}
    test = {0: 0.8774, 3: 0.8610}
    assert val[3] > val[0], "validation preferred round 3"
    assert test[3] < test[0], "test disagreed"
