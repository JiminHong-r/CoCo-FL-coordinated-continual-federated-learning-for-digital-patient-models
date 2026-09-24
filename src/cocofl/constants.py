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

    # Hospital admission and discharge
    ADMISSION = "HOSPITAL_ADMISSION"
    DISCHARGE = "HOSPITAL_DISCHARGE"

    # MIMIC only
    ICU_ADMISSION = "ICU_ADMISSION"
    ICU_DISCHARGE = "ICU_DISCHARGE"
    ED_ADMISSION = "ED_REGISTRATION"
    ED_DISCHARGE = "ED_OUT"
    SOFA = "SOFA"


STATIC_DATA_FN = "static_data.pickle"


# Time-limited labels. A task listed here is positive only if its outcome happens
# within the horizon of the prediction point; a later event counts as negative.
# Tasks not listed use the next qualifying event, whenever it happens.
#
# readmission uses 30 days, the standard clinical definition. hospital_mortality and
# icu_admission are already limited by the current stay (their outcome search stops
# at discharge), so they are not listed.
TASK_HORIZONS: dict[str, timedelta] = {
    "readmission": timedelta(days=30),
}

# ``true_token_time`` on every InferenceDataset sample is ``times[outcome_idx] -
# times[start_idx]`` in microseconds, so horizons use the same unit.
MICROSECONDS_PER_SECOND = 1_000_000


def horizon_microseconds(task_name: str) -> float | None:
    """Label horizon for ``task_name`` in microseconds, or ``None`` if unbounded."""
    horizon = TASK_HORIZONS.get(task_name)
    if horizon is None:
        return None
    return horizon.total_seconds() * MICROSECONDS_PER_SECOND
