from datetime import timedelta
from enum import StrEnum
from importlib.resources import files
from pathlib import Path

MAPPINGS_DIR: Path = files("cocofl.preprocess.maps")
PROJECT_ROOT: Path = (MAPPINGS_DIR / "../../../..").resolve()


class SpecialToken(StrEnum):
    DOB = "MEDS_BIRTH"
    DEATH = "MEDS_DEATH"
    TIMELINE_END = "TIMELINE_END"

    # hospital adm and dc
    ADMISSION = "HOSPITAL_ADMISSION"
    DISCHARGE = "HOSPITAL_DISCHARGE"

    # only MIMIC
    ICU_ADMISSION = "ICU_ADMISSION"
    ICU_DISCHARGE = "ICU_DISCHARGE"
    ED_ADMISSION = "ED_REGISTRATION"
    ED_DISCHARGE = "ED_OUT"
    SOFA = "SOFA"


STATIC_DATA_FN = "static_data.pickle"


# ---- Time-bounded downstream labels ----------------------------------------
# A task listed here is POSITIVE only when its outcome token occurs within the
# horizon of the prediction point; a qualifying event *beyond* the horizon is a
# NEGATIVE. Tasks absent from this mapping stay unbounded ("the next qualifying
# event, whenever it happens").
#
# ``readmission`` is bounded to 30 days so it matches the standard clinical
# definition of 30-day hospital readmission rather than "re-admitted ever".
#
# Note this is only meaningful for tasks whose outcome search can actually run
# past the horizon. ``hospital_mortality`` and ``icu_admission`` are already
# bounded by the index stay (their outcome search terminates at the stay's
# discharge), so adding them here would change little.
TASK_HORIZONS: dict[str, timedelta] = {
    "readmission": timedelta(days=30),
}

# ``true_token_time`` on every InferenceDataset sample is
# ``times[outcome_idx] - times[start_idx]`` in MICROSECONDS, so horizons are
# compared in the same unit.
MICROSECONDS_PER_SECOND = 1_000_000


def horizon_microseconds(task_name: str) -> float | None:
    """Label horizon for ``task_name`` in microseconds, or ``None`` if unbounded."""
    horizon = TASK_HORIZONS.get(task_name)
    if horizon is None:
        return None
    return horizon.total_seconds() * MICROSECONDS_PER_SECOND
