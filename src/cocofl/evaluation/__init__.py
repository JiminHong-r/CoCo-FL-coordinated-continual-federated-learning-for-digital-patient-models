"""Comprehensive evaluation harness.

Axes covered:
- Downstream predictive performance (per centre + macro)
- Personalisation gain (Δ_cohort, Δ_patient)
- Catastrophic forgetting (BWT, FWT, population stability across rounds)
- Upward propagation effectiveness (cohort improvement vs top-down-only)
- Robustness to heterogeneity (α sweep)
- System/communication efficiency (params/round, total cost)

Baselines:
- Population-only      : frozen population, no adapters
- Local fine-tune      : per-centre + per-patient fine-tune without coordination
- Standard FedAvg      : flat FedAvg adapter averaging (no hierarchy)
- Top-down only        : hierarchical init from cohort, no upward propagation
"""
from .metrics import (
    AucPerLevel,
    delta_cohort,
    delta_patient,
    retention,
    bwt,
    fwt,
    auprc,
    expected_calibration_error,
    macro_average,
    variance,
    communication_bytes,
    naive_full_model_bytes,
    total_communication_cost,
    comm_savings_ratio,
)
from .baselines import (
    eval_population_only,
    run_local_finetune,
    run_top_down_only,
    run_standard_fedavg,
)

__all__ = [
    "AucPerLevel",
    "delta_cohort",
    "delta_patient",
    "retention",
    "bwt",
    "fwt",
    "auprc",
    "expected_calibration_error",
    "macro_average",
    "variance",
    "communication_bytes",
    "eval_population_only",
    "run_local_finetune",
    "run_top_down_only",
    "run_standard_fedavg",
]
