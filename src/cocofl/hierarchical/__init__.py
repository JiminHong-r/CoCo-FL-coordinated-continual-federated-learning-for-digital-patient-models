"""CoCo-FL hierarchical 3-level adaptation (Spec WP4).

Three nested levels of adaptation on top of a frozen population backbone:

    Population (frozen)
        └ Cohort adapter (per clinical centre, medium pace)
              └ Patient adapter (per patient, fast pace)

Adaptation flows BIDIRECTIONALLY:
- Top-down: cohort init from population, patient init from cohort
- Bottom-up: patient deltas aggregate into cohort updates; cohort updates
  aggregate into slow population evolution (``upward_prop`` module).
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
