"""Dirichlet partitioning of patients across simulated hospitals (WP2/WP4).

Adapted from ``fl-ight/data_processing/data_partitioning.py`` (Carlos's branch). The
original groups patients by diagnosis category and draws a separate Dirichlet over
hospitals for each class, so alpha controls how non-IID the hospitals are. Here the
class is whatever label the caller supplies (the class-aware split uses age
tertiles); without labels, one Dirichlet is drawn over all patients.

Lower alpha: more skewed (one hospital dominates a class). Higher alpha: close to
uniform.

Use ``HospitalSplit.bind_to_dataset`` to get per-hospital index lists for a dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class HospitalSplit:
    """Which patients, and which dataset indices, belong to each hospital.

    Attributes ---------- num_hospitals : int alpha : float Dirichlet concentration;
    small = skewed, large = uniform. pids_per_hospital : dict[int, list[int]]
    Hospital id -> patient ids. indices_per_hospital : dict[int, list[int]] Hospital
    id -> dataset indices (filled by ``bind_to_dataset``). class_labels : dict[int,
    str] | None Patient -> class label used for the per-class draws, if any.
    """
    num_hospitals: int
    alpha: float
    pids_per_hospital: dict[int, list[int]]
    indices_per_hospital: dict[int, list[int]] = field(default_factory=dict)
    class_labels: dict[int, str] | None = None

    def bind_to_dataset(self, dataset) -> "HospitalSplit":
        """Fill ``indices_per_hospital`` from the dataset's patient ids.

        ``patient_id_at_idx`` is keyed by token index. A task dataset
        (``InferenceDataset``) is indexed by prediction point instead: sample ``i``
        sits at token ``start_indices[i]``, so that is the key to look up. Using
        ``i`` directly would ask "who owns token i", a different patient, and would
        put nearly every window in the wrong place.

        ``TimelineDataset`` is indexed by token, so it has no ``start_indices`` and
        the sample index is already the right key.
        """
        pid_set_per_hospital = {h: set(pids) for h, pids in self.pids_per_hospital.items()}
        indices = {h: [] for h in pid_set_per_hospital}
        pid_at_idx = dataset.patient_id_at_idx
        starts = getattr(dataset, "start_indices", None)

        if starts is not None:
            # One batched lookup instead of one call per window (the accessor is
            # lazy on the sharded backend). Same approach as
            # personalisation.loop.per_patient_subsets.
            import torch

            keys = torch.as_tensor(starts, dtype=torch.long)
            pids = pid_at_idx[keys].reshape(-1).tolist()
        else:
            pids = (int(pid_at_idx[i]) for i in range(len(dataset)))

        for idx, pid in enumerate(pids):
            pid = int(pid)
            for h, pset in pid_set_per_hospital.items():
                if pid in pset:
                    indices[h].append(idx)
                    break
        self.indices_per_hospital = indices
        return self


def dirichlet_split_patients(
    patient_ids: Iterable[int],
    num_hospitals: int,
    alpha: float,
    class_labels: dict[int, str] | None = None,
    seed: int = 42,
) -> HospitalSplit:
    """Split patients across hospitals with a Dirichlet draw.

    Parameters ---------- patient_ids Unique patient ids to distribute.
    num_hospitals Number of hospitals. alpha Dirichlet concentration. Small (<1) =
    highly skewed. class_labels Optional pid -> class label. If given, a separate
    Dirichlet is drawn per class, so each hospital gets a different mix of classes.
    If None, one Dirichlet is drawn over all patients (hospitals differ only in
    size). seed Numpy RNG seed.
    """
    rng = np.random.default_rng(seed)
    pids = list(patient_ids)

    if class_labels is not None:
        # Group by class, then one Dirichlet per class
        classes: dict[str, list[int]] = {}
        for p in pids:
            cls = class_labels.get(p)
            if cls is None:
                continue
            classes.setdefault(cls, []).append(p)

        hospital_pids: dict[int, list[int]] = {h: [] for h in range(num_hospitals)}
        for cls, cls_pids in classes.items():
            rng.shuffle(cls_pids)
            props = rng.dirichlet([alpha] * num_hospitals)
            sizes = (props * len(cls_pids)).astype(int)
            # Give the rounding remainder to the last hospital so every patient is
            # assigned.
            sizes[-1] = len(cls_pids) - sizes[:-1].sum()
            start = 0
            for h, sz in enumerate(sizes):
                hospital_pids[h].extend(cls_pids[start:start + sz])
                start += sz
    else:
        # No class labels: one Dirichlet over all patients
        rng.shuffle(pids)
        props = rng.dirichlet([alpha] * num_hospitals)
        sizes = (props * len(pids)).astype(int)
        sizes[-1] = len(pids) - sizes[:-1].sum()
        hospital_pids = {}
        start = 0
        for h, sz in enumerate(sizes):
            hospital_pids[h] = pids[start:start + sz]
            start += sz

    return HospitalSplit(
        num_hospitals=num_hospitals,
        alpha=alpha,
        pids_per_hospital=hospital_pids,
        class_labels=class_labels,
    )


def iid_split_patients(
    patient_ids: Iterable[int],
    num_hospitals: int,
    seed: int = 42,
) -> HospitalSplit:
    """Uniform random split, the IID reference."""
    rng = np.random.default_rng(seed)
    pids = list(patient_ids)
    rng.shuffle(pids)
    sizes = [len(pids) // num_hospitals] * num_hospitals
    sizes[-1] += len(pids) - sum(sizes)
    hospital_pids: dict[int, list[int]] = {}
    start = 0
    for h, sz in enumerate(sizes):
        hospital_pids[h] = pids[start:start + sz]
        start += sz
    return HospitalSplit(num_hospitals=num_hospitals, alpha=float("inf"),
                         pids_per_hospital=hospital_pids)


def summarize_split(split: HospitalSplit) -> str:
    """Print the number of patients (and per-class counts, if any) per hospital."""
    lines = [f"HospitalSplit: {split.num_hospitals} hospitals, alpha={split.alpha}"]
    for h, pids in sorted(split.pids_per_hospital.items()):
        line = f"  hosp {h}: {len(pids):,} patients"
        if split.indices_per_hospital:
            line += f"  ({len(split.indices_per_hospital.get(h, [])):,} samples)"
        if split.class_labels:
            counts: dict[str, int] = {}
            for p in pids:
                cls = split.class_labels.get(p, "?")
                counts[cls] = counts.get(cls, 0) + 1
            line += "  " + " ".join(f"{c}={n}" for c, n in sorted(counts.items()))
        lines.append(line)
    return "\n".join(lines)
