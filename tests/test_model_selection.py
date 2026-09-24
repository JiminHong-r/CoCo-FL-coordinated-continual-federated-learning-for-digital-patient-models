"""Adaptation must be reported at its best validation round, not its last.

``EarlyStoppingMonitor`` only decides when to stop; it never keeps a best state.
With ``patience=999`` and 5 rounds it never stops early, so without selection every
method would report round 5, even when drift made it worse than not adapting.
Keeping the best validation round, with round 0 (the population model) as a
candidate, means adaptation is never reported below its starting point on
validation.
"""
from __future__ import annotations

import torch

from cocofl.evaluation.baselines import _snapshot
from cocofl.evaluation.early_stopping import EarlyStoppingMonitor


def _select_best(val_by_round):
    """Same selection as the harness: round 0 included, strict improvement only."""
    best_val, best_round = val_by_round[0], 0
    for r in range(1, len(val_by_round)):
        v = val_by_round[r]
        if v is not None and (best_val is None or v > best_val):
            best_val, best_round = v, r
    return best_val, best_round


def test_degrading_run_falls_back_to_round_zero():
    """Every round is worse than not adapting, so round 0 must be chosen."""
    best_val, best_round = _select_best([0.877, 0.860, 0.855, 0.851, 0.848, 0.845])
    assert best_round == 0
    assert best_val == 0.877, "must fall back to the population model"


def test_improving_run_keeps_the_best_round():
    best_val, best_round = _select_best([0.877, 0.881, 0.892, 0.888, 0.884, 0.880])
    assert best_round == 2 and best_val == 0.892, "peak round, not the final one"


def test_last_round_is_not_what_gets_reported():
    """The last round must not be reported automatically."""
    vals = [0.877, 0.910, 0.870, 0.860, 0.850, 0.840]
    best_val, _ = _select_best(vals)
    assert best_val != vals[-1]
    assert best_val == 0.910


def test_none_validation_scores_are_skipped():
    best_val, best_round = _select_best([0.80, None, 0.85, None])
    assert best_round == 2 and best_val == 0.85


def test_early_stopping_monitor_alone_cannot_select():
    """Shows why a separate snapshot is needed: the monitor keeps no state."""
    m = EarlyStoppingMonitor(patience=999, min_rounds=5)
    for r, v in enumerate([0.9, 0.8, 0.7, 0.6, 0.5], start=1):
        assert m.update(r, v) is False, "patience=999 can never stop the loop"
    assert not hasattr(m, "best_state"), (
        "the monitor tracks only a scalar; restoring a model needs a snapshot"
    )


def test_snapshot_is_a_deep_copy():
    states = {0: {"a": torch.ones(3)}}
    snap = _snapshot(states)
    states[0]["a"] += 1
    assert torch.allclose(snap[0]["a"], torch.ones(3)), (
        "a shallow copy would let later rounds mutate the saved best state"
    )


def test_every_method_selects_on_validation():
    import inspect

    import cocofl.evaluation.baselines as b
    import cocofl.evaluation.run_evaluation as r

    bs, rs = inspect.getsource(b), inspect.getsource(r)
    # local_finetune and top_down_only share _round_loop_with_early_stopping
    assert "coord.cohort_states = best_states" in bs
    assert "server_state = best_state" in bs, "FedAvg must select too"
    assert "coord.cohort_states = _best_states" in rs, "hierarchical must select too"
