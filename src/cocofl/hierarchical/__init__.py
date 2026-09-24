"""CoCo-FL three-level adaptation (WP4).

Two adapter levels on top of a frozen population backbone::

    Population (frozen)
        └ Cohort adapter (per clinical centre, medium pace)
              └ Patient adapter (per patient, fast pace)

Adaptation flows both ways. Downward: a cohort starts from the population and a
patient from their cohort. Upward: patient updates fold into the cohort, and cohort
updates fold slowly into the population (``upward_prop``).
"""
from .cohort import (
    CohortConfig,
    train_cohort_adapter,
    train_all_cohorts,
)
from .patient import (
    PatientConfig,
    train_patient_adapter,
    aggregate_patient_deltas,
)
from .upward_prop import (
    weighted_state_average,
    propagate_patients_to_cohort,
    propagate_cohorts_to_population,
)
from .coordinator import HierarchicalCoordinator, CoordinatorConfig

__all__ = [
    "CohortConfig",
    "train_cohort_adapter",
    "train_all_cohorts",
    "PatientConfig",
    "train_patient_adapter",
    "aggregate_patient_deltas",
    "weighted_state_average",
    "propagate_patients_to_cohort",
    "propagate_cohorts_to_population",
    "HierarchicalCoordinator",
    "CoordinatorConfig",
]
