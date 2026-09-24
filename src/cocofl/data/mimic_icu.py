from pathlib import Path

import torch as th

from ..constants import SpecialToken as ST
from .base import InferenceDataset


class ICUAdmissionDataset(InferenceDataset):
    """Generates patient timelines ending at the HOSPITAL ADMISSION token. The primary target is to
    predict whether the patient was admitted to the ICU.

    Note:
    - Each death occurring outside the ICU is treated as a positive case, based on the assumption
      that the patient should have been admitted to the ICU.
    """

    def __init__(self, input_dir: str | Path, n_positions: int = 2048, **kwargs):
        super().__init__(input_dir, n_positions, **kwargs)
        self.stop_stokens = [ST.ICU_ADMISSION, ST.DISCHARGE] + self.stop_stokens
        self.adm_indices = self._get_indices_of_stokens(ST.ADMISSION)
        # Prediction point = admission token + 2 (the admission token takes 3
        # slots). Stored as start_indices like in every other task dataset, because
        # code that maps a sample back to its token (patient filters,
        # length-bucketed batching) looks for that name.
        self.start_indices = self.adm_indices + 2

        icu_adm_or_dc_or_dth_indices = self._get_indices_of_stokens(
            [ST.DISCHARGE, ST.ICU_ADMISSION, ST.DEATH]
        )
        self.outcome_indices = self._match(icu_adm_or_dc_or_dth_indices, self.adm_indices)

    def __len__(self) -> int:
        return len(self.adm_indices)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        start_idx = self.start_indices[idx]
        outcome_idx = self.outcome_indices[idx]

        return super().__getitem__(start_idx), {
            "expected": self.vocab.decode(self.tokens[outcome_idx]),
            "true_token_dist": (outcome_idx - start_idx).item(),
            "true_token_time": (self.times[outcome_idx] - self.times[start_idx]).item(),
            "icu_stay_id": self._get_icu_stay_id(outcome_idx),
            "patient_id": self.patient_id_at_idx[start_idx].item(),
            "prediction_time": self.times[start_idx].item(),
            "data_idx": start_idx.item(),
        }
