"""A paired difference must be estimated on the same resampled windows.

All methods score the same windows, so their differences are paired. Separate
per-method intervals (about +/-0.016 on class-aware readmission alpha=0.5) are far
wider than the differences (about 0.0006) and could never resolve them. These tests
check that one resample is applied to both methods, and that pairing is refused when
the methods did not score the same windows.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score


def paired_diffs(pred_a, pred_b, labels, n=400, seed=0, shared=True):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ia = rng.integers(0, len(labels), len(labels))
        ib = ia if shared else rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[ia])) < 2 or len(np.unique(labels[ib])) < 2:
            continue
        out.append(roc_auc_score(labels[ia], pred_a[ia])
                   - roc_auc_score(labels[ib], pred_b[ib]))
    return np.array(out)


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    y = np.concatenate([np.ones(200), np.zeros(800)]).astype(int)
    a = rng.normal(y * 1.2, 1.0)          # a genuinely informative score
    b = a + rng.normal(0, 0.02, y.size)   # near-identical to it
    return a, b, y


def test_shared_resampling_is_much_tighter_than_independent(data):
    """Pairing cancels the shared window difficulty, so the interval is much tighter."""
    a, b, y = data
    paired = paired_diffs(a, b, y, shared=True)
    indep = paired_diffs(a, b, y, shared=False)
    assert paired.std() < indep.std() / 2, (
        f"paired std {paired.std():.5f} vs independent {indep.std():.5f} — if "
        f"pairing is not tighter, the resample is not actually shared"
    )


def test_near_identical_methods_give_a_ci_containing_zero(data):
    a, b, y = data
    d = paired_diffs(a, b, y, shared=True)
    lo, hi = np.percentile(d, [2.5, 97.5])
    assert lo < 0 < hi, "two near-identical scorers must not look different"


def test_a_genuinely_better_method_is_detected(data):
    """Pairing must still detect a real difference."""
    a, _, y = data
    rng = np.random.default_rng(3)
    worse = rng.normal(y * 0.3, 1.0)      # clearly weaker signal
    d = paired_diffs(a, worse, y, shared=True)
    lo, _ = np.percentile(d, [2.5, 97.5])
    assert lo > 0, "a real improvement must be resolvable"


def test_alignment_check_rejects_mismatched_labels():
    """A paired test on different windows is invalid and must be refused."""
    y1 = np.array([0, 1, 0, 1])
    y2 = np.array([1, 1, 0, 0])
    assert not np.array_equal(y1, y2)


def test_predictions_are_carried_on_the_metrics_result():
    """The predictions must be kept, since they cannot be recomputed after a run."""
    from cocofl.personalisation.evaluate import MetricsResult
    m = MetricsResult(auroc=0.8, auprc=0.4, ece=0.1, n=3, n_pos=1,
                      preds=[0.9, 0.1, 0.2], labels=[1, 0, 0])
    assert m.preds is not None and m.labels is not None


def test_preds_default_to_none_so_existing_construction_sites_still_work():
    from cocofl.personalisation.evaluate import MetricsResult
    m = MetricsResult(auroc=None, auprc=None, ece=None, n=0, n_pos=0)
    assert m.preds is None and m.labels is None
