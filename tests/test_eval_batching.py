"""Length-bucketed eval batching must not change a single prediction.

The WP4/WP5 harness ran evaluation at batch_size=1, which capped every
hospital's eval set at 150 windows for cost reasons. Batching is what lifts
that cap — but only if it is provably prediction-preserving, because the
downstream sequences are variable-length and the backbone is a recurrence that
would consume any padding we introduced.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset

from cocofl.evaluation.batching import (
    LengthBucketBatchSampler, make_eval_loader, sequence_lengths,
)
from cocofl.evaluation.metrics import auroc_ci95_halfwidth
from cocofl.personalisation.evaluate import predict


CONTEXT_SIZE = 4
TIMELINE_SIZE = 16


class FakeInferenceDataset:
    """Mimics ``InferenceDataset`` + a task dataset's index layout.

    Two patients with interleaved token ranges; window length is clipped both
    by the patient's own history start and by ``timeline_size``, which is what
    makes real downstream windows variable-length.
    """

    def __init__(self):
        self.context_size = CONTEXT_SIZE
        self.timeline_size = TIMELINE_SIZE
        n_tokens = 60
        self.tokens = torch.arange(1, n_tokens + 1)
        self.times = torch.arange(n_tokens) * 1000
        # patient 0 owns [0, 30), patient 1 owns [30, 60)
        self.patient_id_at_idx = torch.tensor([0] * 30 + [1] * 30)
        self.patient_offset_at_idx = torch.tensor([0] * 30 + [30] * 30)
        # Prediction points: some short (near a patient's start), some clipped
        # by timeline_size.
        self.start_indices = torch.tensor([2, 9, 25, 31, 40, 55, 5, 33])

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = int(self.start_indices[idx])
        data_start = int(self.patient_offset_at_idx[start])
        if start - data_start + 1 > self.timeline_size:
            data_start = start + 1 - self.timeline_size
        ctx = torch.full((self.context_size,), 99, dtype=torch.long)
        timeline = self.tokens[data_start:start + 1]
        x = torch.cat((ctx, timeline))
        y = {
            "expected": "POS" if idx % 2 else "NEG",
            "patient_id": int(self.patient_id_at_idx[start]),
            "prediction_time": int(self.times[start]),
            "true_token_time": 0,
        }
        return x, y


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs = torch.stack([b[0] for b in batch])
    n = len(batch)
    return (
        (xs, torch.ones_like(xs, dtype=torch.float32),
         torch.zeros(n, 3, dtype=torch.long), torch.zeros(n, 5)),
        {"expected": [b[1]["expected"] for b in batch]},
    )


def label_fn(y):
    return [1 if e == "POS" else 0 for e in y["expected"]]


class DecayRecurrence(torch.nn.Module):
    """h_t = 0.5*h_{t-1} + x_t, scored at the last timestep.

    Deliberately order- and length-sensitive: a left-pad would leak into the
    state before the real tokens start and shift the output, so this test fails
    loudly if batching ever introduces padding.
    """

    def forward(self, input_ids, elapsed_times=None,
                static_categorical=None, static_numerical=None):
        x = input_ids.to(torch.float32)
        h = torch.zeros(x.size(0))
        for t in range(x.size(1)):
            h = 0.5 * h + x[:, t]
        return (h / 100.0) - 3.0


def test_sequence_lengths_match_getitem():
    ds = FakeInferenceDataset()
    idxs = list(range(len(ds)))
    lengths = sequence_lengths(ds, idxs)
    assert lengths is not None
    actual = [ds[i][0].numel() for i in idxs]
    assert lengths.tolist() == actual


def test_lengths_are_genuinely_variable():
    """Guards the test itself: a fixture where every window is the same length
    would make the batching test vacuous."""
    ds = FakeInferenceDataset()
    lengths = sequence_lengths(ds, range(len(ds)))
    assert len(set(lengths.tolist())) > 1


class LazyLookup:
    """Stands in for the sharded backend's ``LookupData``.

    Not an array: every access costs a search, and there is no cheap way to
    materialise the whole thing. Counts accesses and rejects the full walk, so a
    regression back to coercing the entire object is a test failure rather than
    an hour of silent stalling on the real tree.
    """

    def __init__(self, values):
        self._values = values
        self.accesses = 0

    def __len__(self):
        return len(self._values)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor) and idx.ndim > 0:
            self.accesses += idx.numel()
            return torch.stack([self._values[int(i)] for i in idx])
        if isinstance(idx, (list, tuple)):
            self.accesses += len(idx)
            return torch.stack([self._values[int(i)] for i in idx])
        self.accesses += 1
        if self.accesses > 10_000:
            raise AssertionError("walked the lazy lookup element by element")
        return self._values[int(idx)]


def test_sequence_lengths_does_not_materialise_a_lazy_offset_lookup():
    ds = FakeInferenceDataset()
    n_windows = len(ds)
    ds.patient_offset_at_idx = LazyLookup(ds.patient_offset_at_idx)
    lengths = sequence_lengths(ds, range(n_windows))
    assert lengths is not None
    assert lengths.tolist() == [ds_item.numel() for ds_item in
                                (FakeInferenceDataset()[i][0] for i in range(n_windows))]
    # One access per requested window — never per token in the corpus.
    assert ds.patient_offset_at_idx.accesses == n_windows


def test_sequence_lengths_returns_none_for_unknown_dataset():
    class Opaque:
        def __len__(self):
            return 1

    assert sequence_lengths(Opaque(), [0]) is None


def test_bucket_sampler_covers_every_index_exactly_once():
    ds = FakeInferenceDataset()
    lengths = sequence_lengths(ds, range(len(ds)))
    sampler = LengthBucketBatchSampler(lengths, batch_size=3)
    flat = [i for batch in sampler for i in batch]
    assert sorted(flat) == list(range(len(ds)))
    assert len(sampler) == len(sampler.batches)


def test_every_batch_is_length_homogeneous():
    ds = FakeInferenceDataset()
    lengths = sequence_lengths(ds, range(len(ds)))
    sampler = LengthBucketBatchSampler(lengths, batch_size=8)
    for batch in sampler:
        assert len({int(lengths[i]) for i in batch}) == 1
        assert len(batch) <= 8


def test_batched_predictions_identical_to_unbatched():
    ds = FakeInferenceDataset()
    model = DecayRecurrence()
    device = torch.device("cpu")
    idxs = list(range(len(ds)))

    baseline = DataLoader(Subset(ds, idxs), batch_size=1, shuffle=False,
                          collate_fn=collate)
    ref_preds, ref_labels = predict(model, baseline, label_fn, device)

    for batch_size in (2, 4, 8, 64):
        loader = make_eval_loader(ds, idxs, collate, batch_size=batch_size)
        preds, labels = predict(model, loader, label_fn, device)
        # Bucketing reorders, so compare as (pred, label) multisets.
        got = sorted(zip(preds, labels))
        want = sorted(zip(ref_preds, ref_labels))
        assert len(got) == len(want)
        for (p, y), (rp, ry) in zip(got, want):
            assert y == ry
            assert abs(p - rp) < 1e-6, f"batch_size={batch_size} changed a prediction"


def test_batch_size_one_is_a_plain_loader():
    ds = FakeInferenceDataset()
    loader = make_eval_loader(ds, range(len(ds)), collate, batch_size=1)
    assert loader.batch_size == 1
    assert len(list(loader)) == len(ds)


def test_auroc_ci_reproduces_the_small_sample_regime():
    # 150 windows at ~2 % prevalence (hospital_mortality) => ~3 positives.
    wide = auroc_ci95_halfwidth(0.6, n=150, n_pos=3)
    assert 0.30 < wide < 0.38
    # Same hospital's full test partition.
    tight = auroc_ci95_halfwidth(0.6, n=12383, n_pos=259)
    assert 0.03 < tight < 0.045
    assert wide / tight > 8


def test_auroc_ci_undefined_for_single_class_or_missing_auc():
    assert auroc_ci95_halfwidth(None, 150, 3) is None
    assert auroc_ci95_halfwidth(0.6, 150, 0) is None
    assert auroc_ci95_halfwidth(0.6, 150, 150) is None
