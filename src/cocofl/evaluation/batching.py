"""Length-bucketed batching for downstream evaluation.

Why this exists
---------------
The WP4/WP5 harness evaluated everything at ``batch_size=1``. The backbone is a
T-LSTM: a 2048-step *sequential* recurrence, so a single sequence leaves the GPU
almost entirely idle. Measured cost was ~0.22-0.36 s per window, which is what
forced ``eval_per_hospital=150`` (≈1.2 % of each hospital's test partition) and
made the reported AUROCs small-sample noisy.

Why not just pad
----------------
``InferenceDataset.__getitem__`` returns ``context ++ timeline`` where the
timeline is ``min(patient_history, timeline_size)`` tokens — so sequences are
**variable length** and ``make_collate``'s ``torch.stack`` only works because
the batch size is 1. Left-padding would let us stack, but the T-LSTM recurrence
starts from a zero state and would then consume the pad prefix, perturbing the
cell state before the real tokens begin. That silently changes every number.

So instead we bucket by **exact** sequence length: every batch is homogeneous,
no padding is introduced, and each sequence's recurrence is bit-identical to
what it was at ``batch_size=1`` (the batch dimension is independent in an LSTM).
Where a length has fewer members than ``batch_size`` the batch is simply
smaller, so the worst case degrades gracefully to the old behaviour.

``TFNDownstreamClassifier`` pools ``hidden_states[:, -1, :]``, which stays the
true final timestep for every row precisely because nothing is padded.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset


def sequence_lengths(dataset, indices) -> torch.Tensor | None:
    """Exact input length ``__getitem__`` would produce for each of ``indices``.

    Mirrors ``InferenceDataset.__getitem__``: the timeline is clipped to the
    patient's own history and to ``timeline_size``, and a fixed-width static
    context is prepended::

        len = context_size + min(start - patient_offset[start] + 1, timeline_size)

    Returns ``None`` when the dataset does not expose the attributes needed to
    compute this (callers then fall back to ``batch_size=1``), so an unfamiliar
    dataset degrades to the old behaviour instead of producing wrong batches.

    Cost note: on the sharded backend ``patient_offset_at_idx`` is not an array
    but a lazy ``LookupData`` that runs a ``searchsorted`` per access. It must be
    indexed with exactly the starts we need — coercing the whole object to a
    tensor walks it element by element and takes the better part of an hour on
    the MIMIC tree. Work stays O(len(indices)).
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
    """Yields batches whose members all share one exact sequence length.

    Deterministic: buckets are emitted in ascending length order and members
    keep their original relative order, so a run is reproducible. AUROC/AUPRC/
    ECE are order-invariant, so this does not affect any reported metric.
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
    """Length-homogeneous batches, reshuffled on every pass.

    ``LengthBucketBatchSampler`` is deliberately deterministic because the
    metrics it feeds are order-invariant. Training is not: the optimiser needs a
    different order each round, and it matters more than usual here because
    ``CohortConfig.local_steps_per_epoch`` truncates the loader after N batches.
    Under the deterministic sampler that truncation would hand every round the
    same prefix — the shortest sequences in the partition — and the centre would
    never train on anything else.

    Batches stay exact-length homogeneous, so nothing is padded into the T-LSTM
    recurrence; only the members within a length bucket and the order of the
    batches are permuted.
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

    With ``batch_size <= 1`` (or a dataset whose lengths cannot be derived) this
    is exactly the plain shuffled ``batch_size=1`` loader the harness used
    before. Above that it batches within equal-length buckets, which is what
    makes training on a full hospital partition affordable — a 2048-step T-LSTM
    recurrence run one sequence at a time leaves the GPU almost entirely idle.

    Note the unit change this implies upstream: ``local_steps_per_epoch`` counts
    optimiser steps, so at batch B a centre sees ``steps * B`` windows per round
    rather than ``steps``.
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
    """Eval DataLoader over ``Subset(dataset, indices)``.

    With ``batch_size <= 1`` (or a dataset whose lengths cannot be derived) this
    is exactly the plain ``batch_size=1`` loader the harness used before.
    Otherwise it batches within equal-length buckets — same predictions, fewer
    kernel launches.
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
