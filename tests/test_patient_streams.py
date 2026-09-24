"""Patient-stream construction: the fast paths must match the slow ones.

``per_patient_subsets`` and ``split_indices_by_time`` now read the dataset's arrays
instead of building a 2048-token window per index. These tests check both give the
same result, since a mismatch would quietly regroup patients.
"""
from __future__ import annotations

import torch

from cocofl.personalisation import loop as loop_mod
from cocofl.personalisation.loop import per_patient_subsets, split_indices_by_time

from test_eval_batching import FakeInferenceDataset


def _slow_per_patient_subsets(dataset, min_samples=1):
    """The original implementation, kept here as the reference."""
    buckets: dict[int, list[int]] = {}
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        buckets.setdefault(int(y["patient_id"]), []).append(idx)
    return {pid: ix for pid, ix in buckets.items() if len(ix) >= min_samples}


def _slow_split_by_time(dataset, indices, train_frac=0.5):
    times = [(dataset[i][1]["prediction_time"], i) for i in indices]
    times.sort()
    n_train = max(1, int(len(times) * train_frac))
    return [i for _, i in times[:n_train]], [i for _, i in times[n_train:]]


def setup_function():
    loop_mod._PATIENT_BUCKET_CACHE.clear()


def test_grouping_does_not_walk_a_lazy_patient_id_lookup():
    """``patient_id_at_idx`` is a lazy lookup on the sharded backend; grouping must index it once with all the start indices."""
    from test_eval_batching import LazyLookup

    ds = FakeInferenceDataset()
    reference = _slow_per_patient_subsets(ds)
    ds.patient_id_at_idx = LazyLookup(ds.patient_id_at_idx)
    assert per_patient_subsets(ds) == reference
    assert ds.patient_id_at_idx.accesses == len(ds.start_indices)


def test_fast_grouping_matches_the_original():
    ds = FakeInferenceDataset()
    assert per_patient_subsets(ds) == _slow_per_patient_subsets(ds)


def test_min_samples_filters_the_cached_grouping():
    ds = FakeInferenceDataset()
    per_patient_subsets(ds)  # populate cache with the unfiltered grouping
    for k in (1, 2, 4, 5):
        assert per_patient_subsets(ds, min_samples=k) == _slow_per_patient_subsets(ds, k)


def test_grouping_is_cached_per_dataset():
    ds = FakeInferenceDataset()
    per_patient_subsets(ds)
    assert id(ds) in loop_mod._PATIENT_BUCKET_CACHE

    calls = {"n": 0}
    original = FakeInferenceDataset.__getitem__

    def counting(self, idx):
        calls["n"] += 1
        return original(self, idx)

    FakeInferenceDataset.__getitem__ = counting
    try:
        per_patient_subsets(ds, min_samples=2)
    finally:
        FakeInferenceDataset.__getitem__ = original
    assert calls["n"] == 0, "cached call re-scanned the dataset"


def test_cache_pins_the_dataset_so_ids_cannot_be_recycled():
    ds = FakeInferenceDataset()
    per_patient_subsets(ds)
    assert loop_mod._PATIENT_BUCKET_CACHE[id(ds)][0] is ds


def test_distinct_datasets_do_not_share_a_cache_entry():
    a, b = FakeInferenceDataset(), FakeInferenceDataset()
    b.start_indices = torch.tensor([2, 9, 25])
    assert per_patient_subsets(a) != per_patient_subsets(b)


def test_array_path_keeps_indices_the_getitem_path_would_drop():
    """The one difference between the paths: ``dataset[idx]`` drops indices whose ``__getitem__`` raises, and the array path cannot see them.

    Harmless here, because the other loaders read the same indices through
    ``__getitem__`` and would fail the run anyway.
    """
    ds = FakeInferenceDataset()
    original = FakeInferenceDataset.__getitem__

    def raising(self, idx):
        if idx == 0:
            raise RuntimeError("unreadable window")
        return original(self, idx)

    FakeInferenceDataset.__getitem__ = raising
    try:
        fast = per_patient_subsets(ds)
        slow = _slow_per_patient_subsets_tolerant(ds)
    finally:
        FakeInferenceDataset.__getitem__ = original

    fast_all = sorted(i for ix in fast.values() for i in ix)
    slow_all = sorted(i for ix in slow.values() for i in ix)
    assert fast_all == list(range(len(ds)))
    assert 0 not in slow_all


def _slow_per_patient_subsets_tolerant(dataset, min_samples=1):
    buckets: dict[int, list[int]] = {}
    for idx in range(len(dataset)):
        try:
            _, y = dataset[idx]
        except Exception:
            continue
        buckets.setdefault(int(y["patient_id"]), []).append(idx)
    return {pid: ix for pid, ix in buckets.items() if len(ix) >= min_samples}


def test_fast_time_split_matches_the_original():
    ds = FakeInferenceDataset()
    for pid, idxs in per_patient_subsets(ds).items():
        assert split_indices_by_time(ds, idxs) == _slow_split_by_time(ds, idxs)


def test_min_samples_four_yields_two_eval_windows():
    """Why min_samples defaults to 60: at min_samples=4 a patient gets exactly 2 eval windows, so their AUROC can only be 0, 0.5 or 1."""
    ds = FakeInferenceDataset()
    ds.start_indices = torch.tensor([1, 2, 3, 4])  # one patient, 4 windows
    (early, late), = [
        split_indices_by_time(ds, idxs)
        for idxs in per_patient_subsets(ds, min_samples=4).values()
    ]
    assert len(early) == 2 and len(late) == 2


def test_raising_min_samples_widens_the_patient_eval_set():
    ds = FakeInferenceDataset()
    ds.start_indices = torch.arange(1, 21)  # one patient, 20 windows
    (early, late), = [
        split_indices_by_time(ds, idxs)
        for idxs in per_patient_subsets(ds, min_samples=20).values()
    ]
    assert len(early) == 10 and len(late) == 10
