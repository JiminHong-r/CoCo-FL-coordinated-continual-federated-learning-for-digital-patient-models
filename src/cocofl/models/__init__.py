"""Model architectures.

* :class:`TFNFullModel` — Time-aware LSTM + static encoder + (optional) notes
  cross-attention. Trained as the population backbone.
* :class:`TFNDownstreamClassifier` — adds a small MLP head on top of the
  population backbone for binary downstream tasks (mortality, readmission, ICU).
* :class:`GPT2LMNoBiasModel` — alternative decoder-only architecture.
"""
from .backbone import TFNFullModel
from .classifier import TFNDownstreamClassifier
from .config import TFN_CONFIG
from .gpt2 import (
    Block,
    CausalSelfAttention,
    GPT2LMNoBiasModel,
    MLP,
    ModelOutput,
)

__all__ = [
    "TFNFullModel",
    "TFNDownstreamClassifier",
    "TFN_CONFIG",
    "Block",
    "CausalSelfAttention",
    "GPT2LMNoBiasModel",
    "MLP",
    "ModelOutput",
]
