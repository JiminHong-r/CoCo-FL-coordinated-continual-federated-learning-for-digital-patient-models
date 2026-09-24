"""Cohort-training batches must be same-length, shuffled and complete.

Windows have different lengths, so plain batching fails above batch 1, and at batch
1 a round covers a tiny part of a centre's data. Two properties make batching safe:

* every batch has one exact sequence length, so nothing is padded into the T-LSTM
(which would run through the padding); * the order changes between passes, because
``local_steps_per_epoch`` stops the loader early and a fixed order would give every
round the same first batches.
"""
from __future__ import annotations


from cocofl.evaluation.batching import (
    ShuffledLengthBucketBatchSampler, make_train_loader, sequence_lengths,
)
from tests.test_eval_batching import CONTEXT_SIZE, FakeInferenceDataset, collate


def _lengths(ds):
    return sequence_lengths(ds, range(len(ds)))


def test_every_batch_has_one_exact_length():
    ds = FakeInferenceDataset()
    lengths = _lengths(ds)
    sampler = ShuffledLengthBucketBatchSampler(lengths, batch_size=4, seed=0)
    for batch in sampler:
        assert len({int(lengths[i]) for i in batch}) == 1


def test_covers_every_sample_exactly_once():
    ds = FakeInferenceDataset()
    sampler = ShuffledLengthBucketBatchSampler(_lengths(ds), batch_size=3, seed=0)
    seen = sorted(i for batch in sampler for i in batch)
    assert seen == list(range(len(ds)))


def test_len_matches_the_batches_produced():
    sampler = ShuffledLengthBucketBatchSampler(_lengths(FakeInferenceDataset()),
                                               batch_size=3, seed=0)
    assert len(sampler) == len(list(sampler))


def test_order_changes_between_passes():
    """A fixed order with local_steps_per_epoch would train on the same first batches forever."""
    sampler = ShuffledLengthBucketBatchSampler(
        _lengths(FakeInferenceDataset()), batch_size=1, seed=0,
    )
    first = [b for b in sampler]
    second = [b for b in sampler]
    assert first != second, "sampler must reshuffle on each pass"
    assert sorted(i for b in first for i in b) == sorted(i for b in second for i in b)


def test_batch_size_one_matches_the_plain_shuffled_loader():
    ds = FakeInferenceDataset()
    loader = make_train_loader(ds, range(len(ds)), collate, batch_size=1)
    seen = 0
    for batch in loader:
        (x, _e, _c, _n), _y = batch
        assert x.shape[0] == 1
        seen += 1
    assert seen == len(ds)


def test_batched_loader_stacks_without_padding():
    ds = FakeInferenceDataset()
    loader = make_train_loader(ds, range(len(ds)), collate, batch_size=4, seed=0)
    total = 0
    for batch in loader:
        (x, _e, _c, _n), _y = batch
        # Stacking only works because lengths match; every row must be a real
        # window, never a padded one.
        assert x.shape[1] >= CONTEXT_SIZE
        total += x.shape[0]
    assert total == len(ds)
