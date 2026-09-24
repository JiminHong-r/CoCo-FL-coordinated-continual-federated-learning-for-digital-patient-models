"""Load the pre-computed federated splits (WP2 task 4).

``scripts/build_split_v2.py`` writes one JSON file per alpha::

    {
      "alpha": 0.1,
      "num_hospitals": 3,
      "hospitals": {"0": {"train": [...], "val": [...], "test": [...]}, ...},
      ...
    }

Loading these instead of redrawing the Dirichlet split means every experiment uses
exactly the same patients, with no leakage.
"""
from __future__ import annotations

import json
from pathlib import Path

from .hospital_split import HospitalSplit


def load_split(splits_dir: Path | str, alpha: float) -> dict:
    """Load the raw JSON for the given alpha.

    Call :func:`as_hospital_split` on the result to get a ``HospitalSplit``.
    """
    splits_dir = Path(splits_dir)
    candidates = [
        splits_dir / f"alpha_{alpha}.json",
        splits_dir / f"alpha_{float(alpha)}.json",
        splits_dir / f"alpha_{int(alpha) if float(alpha).is_integer() else alpha}.json",
    ]
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text())
    available = sorted(splits_dir.glob("alpha_*.json"))
    raise FileNotFoundError(
        f"No federated split for α={alpha} in {splits_dir}. "
        f"Available: {[p.name for p in available]}"
    )


def as_hospital_split(split_json: dict, partition: str = "train") -> HospitalSplit:
    """Build a :class:`HospitalSplit` for one partition.

    ``partition`` is ``"train"`` (training), ``"val"`` (selection) or ``"test"``
    (evaluation). The result has ``pids_per_hospital`` filled; call
    ``bind_to_dataset(dataset)`` to map patient ids to dataset indices.
    """
    if partition not in ("train", "val", "test"):
        raise ValueError(f"partition must be train/val/test, got {partition!r}")
    pids_per_hospital = {
        int(hid): list(parts[partition])
        for hid, parts in split_json["hospitals"].items()
    }
    return HospitalSplit(
        num_hospitals=split_json["num_hospitals"],
        alpha=split_json["alpha"],
        pids_per_hospital=pids_per_hospital,
    )


def list_available(splits_dir: Path | str) -> list[float]:
    splits_dir = Path(splits_dir)
    return sorted(float(p.stem.split("alpha_")[1])
                  for p in splits_dir.glob("alpha_*.json"))
