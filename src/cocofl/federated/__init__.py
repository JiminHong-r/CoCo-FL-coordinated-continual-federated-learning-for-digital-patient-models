"""Federated learning across simulated hospital silos.

Partitions the tokenized patient cohort into N silos via a Dirichlet
distribution (low alpha => skewed, high alpha => near-IID), trains one FFN
adapter per silo, and averages adapter weights round-by-round (FedAvg).

The population backbone is frozen — only adapter (and optionally classifier
head) parameters travel between silos and server.

This is the flat / non-hierarchical baseline. The hierarchical bidirectional
variant lives in :mod:`cocofl.hierarchical`.
"""
from .hospital_split import (
    dirichlet_split_patients,
    iid_split_patients,
    HospitalSplit,
    summarize_split,
)
from .fixed_splits import (
    load_split,
    as_hospital_split,
    list_available,
)

__all__ = [
    "dirichlet_split_patients",
    "iid_split_patients",
    "HospitalSplit",
    "summarize_split",
    "load_split",
    "as_hospital_split",
    "list_available",
]
