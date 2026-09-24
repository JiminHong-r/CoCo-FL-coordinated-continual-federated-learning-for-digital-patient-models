"""Length-bucketed batching for evaluation and training.

The backbone is a T-LSTM, a 2048-step sequential recurrence, so running one sequence
at a time leaves the GPU mostly idle (about 0.2-0.4 s per window). That made
full-size evaluation too slow.

Padding is not an option. Windows have different lengths (the timeline is
``min(patient_history, timeline_size)`` tokens), and the T-LSTM would run through
any padding before the real tokens, changing every result.

So batches are grouped by exact sequence length. Nothing is padded, and each
sequence's output is identical to a ``batch_size=1`` run (batch rows are independent
in an LSTM). A length with fewer members than ``batch_size`` just gives a smaller
batch. ``TFNDownstreamClassifier`` reads ``hidden_states[:, -1, :]``, which stays
each row's true last step because nothing is padded.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset


def sequence_lengths(dataset, indices) -> torch.Tensor | None:
    """The exact input length ``__getitem__`` would produce for each of ``indices``.

    Mirrors ``InferenceDataset.__getitem__``: the timeline is clipped to the
    patient's own history and to ``timeline_size``, and a fixed-width static context
    is added in front::

        len = context_size + min(start - patient_offset[start] + 1, timeline_size)

    Returns ``None`` if the dataset lacks the needed attributes; callers then fall
    back to ``batch_size=1``.

    On the sharded backend ``patient_offset_at_idx`` is a lazy lookup, so index it
    with only the starts needed. Turning the whole thing into a tensor takes close
    to an hour on MIMIC.
    """
    try:
        start_indices = dataset.start_indices
        offsets = dataset.patient_offset_at_idx
        timeline_size = int(dataset.timeline_size)
        context_size = int(dataset.context_size)
    except AttributeError:
        return None

    idx = torch.as_tensor(list(indices), dtype=torch.long)
    starts = torch.as_tensor(start_indices, dtype=torch.long)[idx]
    try:
        offs = torch.as_tensor(offsets[starts], dtype=torch.long).reshape(-1)
    except (TypeError, IndexError, NotImplementedError):
        return None
    if offs.numel() != starts.numel():
        return None
    span = starts - offs + 1
    return context_size + span.clamp(max=timeline_size)


class LengthBucketBatchSampler(torch.utils.data.Sampler):
    """Batches whose members all have the same sequence length.

    Deterministic: buckets come in ascending length order and members keep their
    order. AUROC, AUPRC and ECE do not depend on order, so metrics are unaffected.
    """

    def __init__(self, lengths: torch.Tensor, batch_size: int):
        self.batches: list[list[int]] = []
        buckets: dict[int, list[int]] = defaultdict(list)
        for pos, length in enumerate(lengths.tolist()):
            buckets[length].append(pos)
        for length in sorted(buckets):
            members = buckets[length]
            for i in range(0, len(members), batch_size):
                self.batches.append(members[i:i + batch_size])

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


class ShuffledLengthBucketBatchSampler(torch.utils.data.Sampler):
    """Same-length batches, reshuffled on every pass (for training).

    Training needs a new order each round. It matters here because
    ``CohortConfig.local_steps_per_epoch`` stops the loader after N batches: with a
    fixed order every round would get the same first batches (the shortest
    sequences). Only the order of members within a bucket and the order of batches
    change; nothing is padded.
    """

    def __init__(self, lengths: torch.Tensor, batch_size: int, seed: int = 42):
        self.buckets: list[list[int]] = []
        by_length: dict[int, list[int]] = defaultdict(list)
        for pos, length in enumerate(lengths.tolist()):
            by_length[length].append(pos)
        self.buckets = [by_length[k] for k in sorted(by_length)]
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self._len = sum(
            (len(m) + batch_size - 1) // batch_size for m in self.buckets
        )

    def __iter__(self):
        batches: list[list[int]] = []
        for members in self.buckets:
            perm = torch.randperm(len(members), generator=self.generator).tolist()
            shuffled = [members[i] for i in perm]
            for i in range(0, len(shuffled), self.batch_size):
                batches.append(shuffled[i:i + self.batch_size])
        order = torch.randperm(len(batches), generator=self.generator).tolist()
        return iter([batches[i] for i in order])

    def __len__(self) -> int:
        return self._len


def make_train_loader(dataset, indices, collate_fn, batch_size: int = 1,
                      seed: int = 42) -> DataLoader:
    """Shuffled training DataLoader over ``Subset(dataset, indices)``.

    With ``batch_size <= 1`` (or when lengths cannot be derived) this is a plain
    shuffled batch-1 loader. Otherwise it batches within same-length buckets, which
    makes training on a full partition affordable.

    ``local_steps_per_epoch`` counts optimiser steps, so at batch B a centre sees
    ``steps * B`` windows per round.
    """
    indices = list(indices)
    subset = Subset(dataset, indices)
    if batch_size <= 1:
        return DataLoader(subset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    lengths = sequence_lengths(dataset, indices)
    if lengths is None:
        return DataLoader(subset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    return DataLoader(
        subset,
        batch_sampler=ShuffledLengthBucketBatchSampler(lengths, batch_size, seed=seed),
        collate_fn=collate_fn,
    )


def make_eval_loader(dataset, indices, collate_fn, batch_size: int = 1) -> DataLoader:
    """Evaluation DataLoader over ``Subset(dataset, indices)``.

    With ``batch_size <= 1`` (or when lengths cannot be derived) this is a plain
    batch-1 loader. Otherwise it batches within same-length buckets: same
    predictions, faster.
    """
    indices = list(indices)
    subset = Subset(dataset, indices)
    if batch_size <= 1:
        return DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    lengths = sequence_lengths(dataset, indices)
    if lengths is None:
        return DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return DataLoader(
        subset,
        batch_sampler=LengthBucketBatchSampler(lengths, batch_size),
        collate_fn=collate_fn,
    )
