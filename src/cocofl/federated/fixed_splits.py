"""Load pre-computed federated splits (Spec WP2 task 4).

Pairs with ``scripts/build_split_v2.py`` which writes one JSON file
per α value:

    {
      "alpha": 0.1,
      "num_hospitals": 3,
      "hospitals": {"0": {"train": [...], "val": [...], "test": [...]}, ...},
      ...
    }

Loading these instead of re-running the Dirichlet draw guarantees every
experiment (population baseline, federated, hierarchical, evaluation harness)
operates on identical patient sets and respects the no-leakage constraint.
"""
from __future__ import annotations

import json
from pathlib import Path

from .hospital_split import HospitalSplit


def load_split(splits_dir: Path | str, alpha: float) -> dict:
    """Load the raw JSON for the given α.

    Returns the parsed dict; downstream code calls
    :func:`as_hospital_split` if it needs the ``HospitalSplit`` object.
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
    """Materialise a :class:`HospitalSplit` for the given partition.

    Use ``partition="train"`` for federated training, ``"val"`` for tuning, and
    ``"test"`` for held-out evaluation. The returned split has
    ``pids_per_hospital`` populated; the caller still calls
    ``bind_to_dataset(dataset)`` to map patient IDs to dataset indices.
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
