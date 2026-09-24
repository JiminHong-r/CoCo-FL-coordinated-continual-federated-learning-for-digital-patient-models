"""Model architectures.

* :class:`TFNFullModel` — Time-aware LSTM + static encoder. Trained as the
  population backbone.
* :class:`TFNDownstreamClassifier` — adds a small MLP head on top of the
  population backbone for binary downstream tasks (mortality, readmission, ICU).
"""
from .backbone import TFNFullModel
from .classifier import TFNDownstreamClassifier

__all__ = [
    "TFNFullModel",
    "TFNDownstreamClassifier",
]
