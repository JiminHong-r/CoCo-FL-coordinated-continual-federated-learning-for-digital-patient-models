"""Result container shared by every WP5 method.

One ``MethodResult`` holds everything the harness needs from a method: metrics per
hospital (AUROC, AUPRC, ECE), the rounds used, wall-clock time, the per-round
validation trajectory, and method-specific extras (e.g. the final server state or
patient AUROCs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MethodResult:
    aucs: dict[int, float | None] = field(default_factory=dict)        # AUROC per hospital
    auprcs: dict[int, float | None] = field(default_factory=dict)
    eces: dict[int, float | None] = field(default_factory=dict)
    # Eval-set size behind each hospital's metrics. An AUROC means little without
    # it: 150 windows at ~2 % prevalence hold about 3 positives, a 95 % CI of about
    # +/-0.34.
    ns: dict[int, int] = field(default_factory=dict)                    # windows scored
    n_pos: dict[int, int] = field(default_factory=dict)                 # positives among them
    n_rounds: int = 0                                                   # rounds actually run
    # Round restored by model selection. 0 means the un-adapted population model won
    # on validation, so this method's numbers equal the baseline's and say nothing
    # about the method. Do not read methods agreeing in such a cell as a finding.
    selected_round: int | None = None
    # L2 distance between the selected adapter state and round 0, summed over
    # hospitals (0 exactly when selected_round == 0). A non-zero value with an
    # unchanged AUROC means the model moved but the eval set could not see it.
    selected_delta_norm: float | None = None
    # Validation macro AUROC of the selected round and of round 0 (round 0 is the
    # un-adapted model, i.e. population_only on validation). Since round 0 is a
    # candidate, val_auroc >= val_auroc_round0 always, so these show how far
    # selection moved, not whether adaptation helps.
    val_auroc: float | None = None
    val_auroc_round0: float | None = None
    # A third surface, for choosing between configurations. Test can't do it
    # (picking the best of N configs on test reports the luckiest draw), and neither
    # can validation (round 0 always keeps it at or above baseline). probe_auroc is
    # measured on a slice of validation that never picks a round.
    probe_auroc: float | None = None
    wall_clock_sec: float = 0.0                                         # training-only wall-clock
    val_history: list[tuple[int, float | None]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)                 # method-specific
