"""TFN-style static feature extraction from ETHOS static_data.pickle.

Maps each patient to:
  - categorical: [gender_idx, race_idx, marital_idx]  (cardinalities 2, 6, 5)
  - numerical:   [bmi_quantile_z, age_z]              (z-scored)

Used by TFNFullModel's StaticEncoder.
"""
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

from .base import TimelineDataset

GENDER_VOCAB = {"GENDER//M": 0, "GENDER//F": 1}
RACE_VOCAB = {
    "RACE//UNKNOWN": 0, "RACE//WHITE": 1, "RACE//BLACK": 2,
    "RACE//ASIAN": 3, "RACE//HISPANIC": 4, "RACE//OTHER": 5,
}
MARITAL_VOCAB = {
    "MARITAL//UNKNOWN": 0, "MARITAL//SINGLE": 1, "MARITAL//MARRIED": 2,
    "MARITAL//DIVORCED": 3, "MARITAL//WIDOWED": 4,
}
BMI_VOCAB = {"BMI//UNKNOWN": 0, **{f"BMI//Q{i}": i for i in range(1, 11)}}

CATEGORICAL_CARDINALITIES = [2, 6, 5]
NUMERICAL_DIM = 2

MICROSECONDS_PER_YEAR = 31_557_600_000_000


def build_static_arrays(static_data: dict) -> tuple[dict, dict]:
    """Returns (cat_by_pid, num_by_pid) torch tensors."""
    cat_by_pid, num_by_pid = {}, {}
    for pid, data in static_data.items():
        gender = GENDER_VOCAB.get(data.get("GENDER", {}).get("code", ["GENDER//F"])[0], 1)
        race = RACE_VOCAB.get(data.get("RACE", {}).get("code", ["RACE//UNKNOWN"])[0], 0)
        marital = MARITAL_VOCAB.get(data.get("MARITAL", {}).get("code", ["MARITAL//UNKNOWN"])[0], 0)
        cat_by_pid[pid] = torch.tensor([gender, race, marital], dtype=torch.long)

        bmi_codes = data.get("BMI", {}).get("code", [])
        bmi_q = BMI_VOCAB.get(bmi_codes[0], 0) if bmi_codes else 0
        bmi_z = (bmi_q - 5.0) / 3.0  # rough z-score (Q1-Q10 centered at 5.5)

        birth = data.get("MEDS_BIRTH", {}).get("time", [None])[0]
        bmi_times = data.get("BMI", {}).get("time", [])
        ref_time = bmi_times[0] if bmi_times else None
        if birth and ref_time:
            age_years = (ref_time - birth) / MICROSECONDS_PER_YEAR
        else:
            age_years = 50.0
        age_z = (age_years - 60.0) / 20.0  # MIMIC adult population centered ~60yo

        num_by_pid[pid] = torch.tensor([bmi_z, age_z], dtype=torch.float32)
    return cat_by_pid, num_by_pid


class TFNTimelineDataset(TimelineDataset):
    """TimelineDataset that also returns elapsed_times and static (cat, num) per window.

    __getitem__ returns: (x, elapsed_times, static_cat, static_num), y
    Compatible with TimelineDataset's existing window construction (context tokens prepended).
    """

    def __init__(self, input_dir, n_positions: int = 2048, **kwargs):
        super().__init__(input_dir, n_positions, **kwargs)
        self._cat_by_pid, self._num_by_pid = build_static_arrays(self.static_data)

    def _elapsed_for_window(self, start_idx: int, length: int) -> torch.Tensor:
        times = self.times[start_idx : start_idx + length].float()
        if times.numel() < 2:
            return torch.zeros(length, dtype=torch.float32)
        deltas = torch.diff(times)
        # microseconds -> hours to keep numbers reasonable for log()
        deltas = (deltas.clamp(min=0) / 3_600_000_000).float()
        elapsed = torch.cat([torch.zeros(1), deltas])
        return elapsed

    def __getitem__(self, idx: int):
        x, y = super().__getitem__(idx)
        pid = int(self.patient_id_at_idx[idx].item())

        cat = self._cat_by_pid.get(pid, torch.zeros(3, dtype=torch.long))
        num = self._num_by_pid.get(pid, torch.zeros(NUMERICAL_DIM, dtype=torch.float32))

        # elapsed_times: length must match x (= context_size + timeline_size)
        # Context tokens are atemporal, set elapsed=1.0 (log-stable)
        timeline_elapsed = self._elapsed_for_window(idx, self.timeline_size)
        context_elapsed = torch.ones(self.context_size, dtype=torch.float32)
        elapsed = torch.cat([context_elapsed, timeline_elapsed])
        # pad/truncate to match x exactly
        if elapsed.numel() < x.numel():
            elapsed = torch.cat([elapsed, torch.ones(x.numel() - elapsed.numel())])
        elif elapsed.numel() > x.numel():
            elapsed = elapsed[: x.numel()]

        return (x, elapsed, cat, num), y


def collate_with_static(batch):
    """Collate (x, elapsed, cat, num), y -> batched tensors."""
    xs = torch.stack([b[0][0] for b in batch])
    elapsed = torch.stack([b[0][1] for b in batch])
    cats = torch.stack([b[0][2] for b in batch])
    nums = torch.stack([b[0][3] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    return (xs, elapsed, cats, nums), ys
