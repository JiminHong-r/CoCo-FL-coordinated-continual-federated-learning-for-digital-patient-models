"""Early stopping helper for federated round loops.

Used by the baselines and the hierarchical runner to decide when to stop the
round loop based on validation-set macro AUC.

Pattern::

    monitor = EarlyStoppingMonitor(patience=3, min_rounds=2)
    for r in range(1, max_rounds + 1):
        train_one_round(...)
        val_macro = macro_average(eval_per_hospital(val_loaders, ...))
        if monitor.update(round_idx=r, val_score=val_macro):
            break    # patience exhausted
    n_rounds_used = monitor.last_round
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EarlyStoppingMonitor:
    patience: int = 3
    min_rounds: int = 2
    best: float = float("-inf")
    stale: int = 0
    last_round: int = 0
    history: list[tuple[int, float | None]] = field(default_factory=list)

    def update(self, round_idx: int, val_score: float | None) -> bool:
        """Record a round; return True if training should stop now."""
        self.last_round = round_idx
        self.history.append((round_idx, val_score))
        if val_score is None:
            # Treat a None score as no improvement instead of failing.
            self.stale += 1
        elif val_score > self.best:
            self.best = float(val_score)
            self.stale = 0
        else:
            self.stale += 1
        return round_idx >= self.min_rounds and self.stale >= self.patience
