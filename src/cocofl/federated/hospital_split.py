"""Dirichlet-based patient partitioning across hospital silos for WP4.

Adapted from ``fl-ight/data_processing/data_partitioning.py`` (Carlos's branch).
The original groups patients by a ``diagnoses_category`` ("cancer", "heart_lung",
"Other") and draws a separate Dirichlet over hospitals per class — yielding
controllable non-IIDness through alpha. We follow the same recipe but the class
labels come from the ETHOS static features (or fall back to plain random
Dirichlet over patients).

Lower alpha → more skewed (one hospital dominates a class).
Higher alpha → near-uniform across hospitals.

Use ``HospitalSplit.from_dataset`` to bind to a TFNTimelineDataset and get
ready-made per-hospital index lists for DataLoader subsetting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class HospitalSplit:
    """Per-hospital assignment of patient IDs and dataset indices.

    Attributes
    ----------
    num_hospitals : int
    alpha : float
        Dirichlet concentration; small => skewed, large => uniform.
    pids_per_hospital : dict[int, list[int]]
        Hospital ID -> assigned patient IDs.
    indices_per_hospital : dict[int, list[int]]
        Hospital ID -> dataset indices (populated by ``bind_to_dataset``).
    class_labels : dict[int, str] | None
        Optional patient -> coarse class label used for the Dirichlet draws
        (when class-aware mode was used).
    """
    num_hospitals: int
    alpha: float
    pids_per_hospital: dict[int, list[int]]
    indices_per_hospital: dict[int, list[int]] = field(default_factory=dict)
    class_labels: dict[int, str] | None = None

    def bind_to_dataset(self, dataset) -> "HospitalSplit":
        """Populate ``indices_per_hospital`` from ``dataset.patient_id_at_idx``.

        ``patient_id_at_idx`` is keyed by **token** index: it searchsorts the
        index against ``patient_offsets``, which are cumulative token counts.

        A task dataset (any ``InferenceDataset``) is indexed by **prediction
        point**, not by token — sample ``i`` lives at token ``start_indices[i]``.
        Looking up ``patient_id_at_idx[i]`` there answers "who owns token i",
        which is a different patient entirely. On the MIMIC cohort the task
        datasets hold ~490 K samples against ~247 M tokens, so ``range(len(ds))``
        only ever reaches the first 0.2 % of the corpus: nearly every window gets
        attributed to one of the few earliest patients, matches no hospital's
        patient set, and is dropped. The partition it produced was arbitrary.

        ``TimelineDataset`` *is* indexed by token, so there ``start_indices`` is
        absent and the sample index is already the right key.
        """
        pid_set_per_hospital = {h: set(pids) for h, pids in self.pids_per_hospital.items()}
        indices = {h: [] for h in pid_set_per_hospital}
        pid_at_idx = dataset.patient_id_at_idx
        starts = getattr(dataset, "start_indices", None)

        if starts is not None:
            # One batched lookup rather than a Python-level call per window;
            # ``patient_id_at_idx`` is a lazy per-index accessor on the sharded
            # backend. Mirrors personalisation.loop.per_patient_subsets.
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
    """Carlos-style Dirichlet partitioning.

    Parameters
    ----------
    patient_ids
        Iterable of unique patient IDs to distribute.
    num_hospitals
        How many silos to create.
    alpha
        Dirichlet concentration (per-class). Small (<1) => highly skewed.
    class_labels
        Optional mapping pid -> coarse class label. When provided, the Dirichlet
        is drawn per-class so per-class shares are heterogeneous across hospitals
        (the original Carlos recipe). When ``None``, a single Dirichlet over all
        patients is drawn (random heterogeneous quantity split).
    seed
        Numpy RNG seed.
    """
    rng = np.random.default_rng(seed)
    pids = list(patient_ids)

    if class_labels is not None:
        # Group by class, then per-class Dirichlet
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
            # ensure all patients are assigned (rounding leftover -> last hospital)
            sizes[-1] = len(cls_pids) - sizes[:-1].sum()
            start = 0
            for h, sz in enumerate(sizes):
                hospital_pids[h].extend(cls_pids[start:start + sz])
                start += sz
    else:
        # No class labels: single Dirichlet over all patients
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
    """Uniform random split — IID baseline for comparison against Dirichlet."""
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
    """Pretty-print number of patients (and per-class counts when available) per hospital."""
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
