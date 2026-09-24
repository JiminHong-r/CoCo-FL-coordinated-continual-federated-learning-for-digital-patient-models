"""Simulated hospitals and the federated runtime.

``hospital_split`` divides patients across hospitals with a Dirichlet draw (low
alpha = skewed, high alpha = close to IID), ``fixed_splits`` loads the pre-computed
splits, and ``flower_app`` runs the federated steps through Flower. Only adapter
parameters (and the head, if trained) travel between hospitals and the server; the
backbone stays frozen. The methods themselves are in
:mod:`cocofl.evaluation.baselines` and :mod:`cocofl.hierarchical`.
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
