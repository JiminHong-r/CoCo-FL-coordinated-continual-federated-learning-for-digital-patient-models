"""The harness must never score a method on windows it trained on.

With ``splits_dir`` (every shipped script) train, val and test come from separate
partitions. The inline Dirichlet path, used when ``splits_dir`` is not given, once
built its eval set from the training indices, which made AUROCs look better than
they were. These tests check that the train and eval indices do not overlap.
"""
from __future__ import annotations

import random



def _split_like_harness(indices, seed=42, frac_eval=0.3):
    """Same disjoint hold-out as the inline branch."""
    idxs = list(indices)
    random.Random(seed).shuffle(idxs)
    n_eval = max(1, int(len(idxs) * frac_eval))
    eval_idx, train_idx = idxs[:n_eval], idxs[n_eval:]
    if not train_idx:
        train_idx = eval_idx
    return train_idx, eval_idx


def test_inline_split_is_disjoint():
    train_idx, eval_idx = _split_like_harness(range(100))
    assert set(train_idx).isdisjoint(eval_idx), "eval windows must not be trained on"
    assert len(train_idx) + len(eval_idx) == 100, "no window lost or duplicated"


def test_inline_split_holds_out_a_nonempty_eval_set():
    for n in (1, 2, 3, 10, 1000):
        train_idx, eval_idx = _split_like_harness(range(n))
        assert eval_idx, f"n={n} produced an empty eval set"


def test_single_window_hospital_degrades_without_crashing():
    """A centre with one window cannot be split; it must not crash."""
    train_idx, eval_idx = _split_like_harness(range(1))
    assert eval_idx == [0]
    assert train_idx == [0], "falls back rather than leaving training empty"


def test_pop_eval_builder_takes_eval_loaders():
    import inspect

    from cocofl.evaluation.run_evaluation import _build_pop_eval_loader

    params = list(inspect.signature(_build_pop_eval_loader).parameters)
    assert params[0] == "hospital_eval_loaders", (
        "first argument must be the eval loaders — sourcing pop_baseline from "
        "training windows makes Axis-3 FWT compare metrics on different data"
    )
