"""The population probe must cope with class imbalance.

The old probe did not crash; it reported 0.589 AUROC on hospital_mortality. It drew
training windows at random (about 60 positives at 2 % prevalence), weighted both
classes equally, and kept the checkpoint with the lowest validation loss, which on
imbalanced data is the one closest to predicting the base rate. Each of those three
is tested here.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cocofl.downstream.probe import (
    fit_balanced_probe, stratified_subsample, window_labels,
)


CONTEXT_SIZE = 4
TIMELINE_SIZE = 16


class FakeTaskDataset:
    """Index layout of an ``InferenceDataset`` task with a rare positive class.

    ``outcome_indices`` point at token positions; a position holding ``POS_TOKEN``
    is a positive outcome, as in the real datasets.
    """

    POS_TOKEN = 7
    NEG_TOKEN = 3

    def __init__(self, n_samples=200, n_pos=8, horizon_ok=None):
        self.context_size = CONTEXT_SIZE
        self.timeline_size = TIMELINE_SIZE
        n_tokens = 4 * n_samples
        self.tokens = torch.full((n_tokens,), self.NEG_TOKEN)
        self.times = torch.arange(n_tokens) * 1000

        self.start_indices = torch.arange(n_samples) * 2
        self.outcome_indices = self.start_indices + 1
        # The first n_pos samples get the positive outcome token.
        self.tokens[self.outcome_indices[:n_pos]] = self.POS_TOKEN
        # Optionally move some positives past a label horizon.
        if horizon_ok is not None:
            for i in range(n_pos):
                dt = 500 if i < horizon_ok else 10_000_000
                self.times[int(self.outcome_indices[i])] = (
                    int(self.times[int(self.start_indices[i])]) + dt
                )

    def __len__(self):
        return len(self.start_indices)

    def _get_indices_of_stokens(self, stokens):
        ids = torch.tensor([self.POS_TOKEN if s == "POS" else self.NEG_TOKEN for s in stokens])
        return torch.nonzero(torch.isin(self.tokens, ids)).view(-1)


def test_window_labels_match_outcome_tokens():
    ds = FakeTaskDataset(n_samples=200, n_pos=8)
    y = window_labels(ds, {"POS"})
    assert y.sum() == 8
    assert y[:8].all() and not y[8:].any()


def test_window_labels_apply_the_label_horizon():
    """An outcome past the horizon is negative, as in label_from_batch."""
    ds = FakeTaskDataset(n_samples=200, n_pos=8, horizon_ok=3)
    y = window_labels(ds, {"POS"}, horizon_us=1000)
    assert y.sum() == 3, "only outcomes inside the horizon stay positive"


def test_stratified_subsample_keeps_every_positive():
    """The stratified draw must keep every positive.

    At 4 % prevalence a random 100-of-500 draw has about 4 positives; the stratified
    draw must have all 20.
    """
    y = np.zeros(500, dtype=np.int8)
    y[:20] = 1
    rng = np.random.default_rng(0)

    sel = stratified_subsample(y, range(len(y)), n_total=100, rng=rng)

    assert len(sel) == 100
    assert y[np.asarray(sel)].sum() == 20


def test_stratified_subsample_caps_positives_at_half_the_budget():
    """A common outcome must not give an all-positive draw."""
    y = np.ones(500, dtype=np.int8)
    y[:50] = 0
    sel = stratified_subsample(y, range(len(y)), n_total=100, rng=np.random.default_rng(0))
    kept = y[np.asarray(sel)]
    assert kept.sum() == 50 and (kept == 0).sum() == 50


def test_stratified_subsample_respects_the_candidate_pool():
    """Only windows inside the given pool (e.g. the reserve) may be drawn."""
    y = np.zeros(500, dtype=np.int8)
    y[:20] = 1
    pool = list(range(0, 250))
    sel = stratified_subsample(y, pool, n_total=100, rng=np.random.default_rng(0))
    assert set(sel) <= set(pool)


def test_balanced_head_beats_the_base_rate_predictor_when_imbalanced():
    """A signal at 2 % prevalence must still be learnable.

    The features carry a real, noisy signal. An unweighted head would just predict
    the majority class; the balanced head has to recover the ranking, so AUROC must
    be well above chance.
    """
    rng = np.random.default_rng(0)
    n, d = 4000, 20
    y = (rng.random(n) < 0.02).astype(np.int8)
    X = rng.normal(size=(n, d))
    X[:, 0] += 1.5 * y  # one informative direction

    X_tr, y_tr = X[:3000], y[:3000]
    X_va, y_va = X[3000:3500], y[3000:3500]
    X_te, y_te = X[3500:], y[3500:]

    clf, _c, val_auc = fit_balanced_probe(X_tr, y_tr, X_va, y_va, seed=0)
    from sklearn.metrics import roc_auc_score

    assert val_auc is not None
    assert roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]) > 0.7


def test_balanced_head_survives_a_single_class_validation_split():
    """An undefined validation AUROC must neither crash selection nor drive it."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (rng.random(200) < 0.3).astype(np.int8)
    clf, _c, val_auc = fit_balanced_probe(X, y, X[:10], np.zeros(10, dtype=np.int8), seed=0)
    assert val_auc is None
    assert clf.predict_proba(X).shape == (200, 2)


def test_filter_by_patients_keys_on_the_token_index(tmp_path):
    """The reserve and global-test filters must map each sample to its own patient.

    ``start_indices[i]`` is a token index, and ``patient_id_at_idx`` must be read
    there, not at ``i``; reading at ``i`` gives an unrelated patient.
    """
    import json

    from cocofl.downstream.finetune import filter_by_patients

    ds = FakeTaskDataset(n_samples=10, n_pos=2)
    # Tokens 0-19 belong to patient 100 and 20+ to patient 200. Samples 0-9 sit at
    # tokens 0, 2, ..., 18, so all of them are patient 100, which only shows if the
    # lookup uses start_indices.
    ds.patient_id_at_idx = torch.tensor([100] * 20 + [200] * (len(ds.tokens) - 20))

    path = tmp_path / "pids.json"
    path.write_text(json.dumps({"patient_ids": [100]}))
    assert filter_by_patients(ds, path) == list(range(10))

    path.write_text(json.dumps({"patient_ids": [200]}))
    assert filter_by_patients(ds, path) == []


def test_filter_by_patients_refuses_a_dataset_without_start_indices(tmp_path):
    import json

    from cocofl.downstream.finetune import filter_by_patients

    ds = FakeTaskDataset(n_samples=4, n_pos=1)
    del ds.start_indices
    path = tmp_path / "pids.json"
    path.write_text(json.dumps({"patient_ids": [1]}))
    with pytest.raises(AttributeError, match="start_indices"):
        filter_by_patients(ds, path)


@pytest.mark.parametrize("n_pos", [1, 5])
def test_window_labels_handles_scarce_positives(n_pos):
    ds = FakeTaskDataset(n_samples=100, n_pos=n_pos)
    assert window_labels(ds, {"POS"}).sum() == n_pos
