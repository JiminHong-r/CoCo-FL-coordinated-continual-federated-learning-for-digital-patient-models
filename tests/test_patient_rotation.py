"""Patient rotation across rounds.

Without rotation the coordinator uses ``patient_streams[:patients_per_round]`` every
round, so later patients are never trained on and the same group feeds the cohort
each time. Rotation moves a window through the whole list at the same cost per
round. Off by default, because it changes the method.
"""
from __future__ import annotations

from cocofl.hierarchical.coordinator import CoordinatorConfig, HierarchicalCoordinator


class FakeCoordinator:
    """Tests the selection logic alone; a real coordinator needs a backbone and loaders, which the selection does not use."""

    def __init__(self, patients_per_round, rotation, n_streams):
        self.cfg = CoordinatorConfig(
            patients_per_round=patients_per_round, patient_rotation=rotation,
        )
        self.round_idx = 0
        self.streams = [f"p{i}" for i in range(n_streams)]

    def round(self):
        self.round_idx += 1
        return HierarchicalCoordinator._patients_this_round(self, self.streams)


def test_default_is_the_legacy_fixed_prefix():
    c = FakeCoordinator(patients_per_round=3, rotation=False, n_streams=10)
    assert [c.round() for _ in range(4)] == [["p0", "p1", "p2"]] * 4


def test_rotation_advances_the_window_each_round():
    c = FakeCoordinator(patients_per_round=3, rotation=True, n_streams=9)
    assert [c.round() for _ in range(3)] == [
        ["p0", "p1", "p2"], ["p3", "p4", "p5"], ["p6", "p7", "p8"],
    ]


def test_rotation_wraps_and_keeps_the_round_size_constant():
    c = FakeCoordinator(patients_per_round=4, rotation=True, n_streams=6)
    rounds = [c.round() for _ in range(4)]
    assert rounds[0] == ["p0", "p1", "p2", "p3"]
    assert rounds[1] == ["p4", "p5", "p0", "p1"]
    assert all(len(r) == 4 for r in rounds), "per-round cost must not change"


def test_rotation_covers_more_patients_than_the_prefix():
    """5 rounds x 20 patients over 50 streams reaches all 50 patients; the fixed prefix reaches 20."""
    fixed = FakeCoordinator(patients_per_round=20, rotation=False, n_streams=50)
    rotating = FakeCoordinator(patients_per_round=20, rotation=True, n_streams=50)
    seen_fixed = {p for _ in range(5) for p in fixed.round()}
    seen_rot = {p for _ in range(5) for p in rotating.round()}
    assert len(seen_fixed) == 20
    assert len(seen_rot) == 50


def test_rotation_is_a_noop_when_the_window_covers_everything():
    for rotation in (False, True):
        c = FakeCoordinator(patients_per_round=5, rotation=rotation, n_streams=3)
        assert [c.round() for _ in range(3)] == [["p0", "p1", "p2"]] * 3


def test_no_patient_repeats_within_a_round():
    c = FakeCoordinator(patients_per_round=4, rotation=True, n_streams=6)
    for _ in range(6):
        batch = c.round()
        assert len(set(batch)) == len(batch)


def test_empty_stream_list_is_safe():
    c = FakeCoordinator(patients_per_round=3, rotation=True, n_streams=0)
    assert c.round() == []
