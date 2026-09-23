"""``bind_to_dataset`` must key patient lookups by token index, not sample index.

``patient_id_at_idx`` searchsorts against ``patient_offsets``, which are
cumulative TOKEN counts — so it answers "who owns token i". A task dataset is
indexed by prediction point instead: sample ``i`` lives at token
``start_indices[i]``. Feeding it ``i`` therefore reads an unrelated patient.

On the real cohort this was not a subtle skew. The task datasets hold ~490 K
samples against ~247 M tokens, so ``range(len(dataset))`` only ever reached the
first 0.2 % of the corpus — nearly every window was attributed to one of a few
early patients, matched no hospital, and was silently dropped from the
partition. These tests pin the indexing so that cannot come back.
"""
from __future__ import annotations

import torch

from cocofl.federated.hospital_split import HospitalSplit


class FakeTaskDataset:
    """Sparse prediction points over a token stream owned by two patients."""

    def __init__(self):
        # Tokens 0..49 belong to patient 100, tokens 50..99 to patient 200.
        self.patient_id_at_idx = torch.tensor([100] * 50 + [200] * 50)
        # Four prediction points, deliberately NOT equal to their sample index:
        # samples 0,1 are patient 100; samples 2,3 are patient 200.
        self.start_indices = torch.tensor([10, 40, 60, 90])

    def __len__(self):
        return len(self.start_indices)


class FakeTimelineDataset:
    """Indexed by token directly — sample index IS the right key."""

    def __init__(self):
        self.patient_id_at_idx = torch.tensor([100] * 3 + [200] * 3)

    def __len__(self):
        return 6


def _split():
    return HospitalSplit(
        num_hospitals=2, alpha=1.0,
        pids_per_hospital={0: [100], 1: [200]},
    )


def test_task_dataset_binds_by_prediction_point():
    split = _split().bind_to_dataset(FakeTaskDataset())
    assert split.indices_per_hospital[0] == [0, 1], "patient 100's windows"
    assert split.indices_per_hospital[1] == [2, 3], "patient 200's windows"


def test_task_dataset_binding_is_not_the_sample_index_lookup():
    """The old code read patient_id_at_idx[i]; here that gives patient 100 for
    every sample, so hospital 1 would come back empty. Guard against it."""
    ds = FakeTaskDataset()
    naive = [int(ds.patient_id_at_idx[i]) for i in range(len(ds))]
    assert naive == [100, 100, 100, 100], "fixture must expose the old failure"

    split = _split().bind_to_dataset(ds)
    assert split.indices_per_hospital[1], "hospital 1 must not be empty"


def test_every_window_is_assigned_exactly_once():
    split = _split().bind_to_dataset(FakeTaskDataset())
    assigned = sorted(i for ix in split.indices_per_hospital.values() for i in ix)
    assert assigned == [0, 1, 2, 3], "no window dropped or double-counted"


def test_timeline_dataset_still_binds_by_token_index():
    """A dataset without start_indices is token-indexed; behaviour unchanged."""
    split = _split().bind_to_dataset(FakeTimelineDataset())
    assert split.indices_per_hospital[0] == [0, 1, 2]
    assert split.indices_per_hospital[1] == [3, 4, 5]


def test_patients_outside_every_hospital_are_dropped():
    ds = FakeTaskDataset()
    split = HospitalSplit(num_hospitals=1, alpha=1.0, pids_per_hospital={0: [200]})
    split.bind_to_dataset(ds)
    assert split.indices_per_hospital[0] == [2, 3]
