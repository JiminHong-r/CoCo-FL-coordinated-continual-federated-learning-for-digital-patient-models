"""Uniform result containers for WP5 baseline / hierarchical method runs.

A single ``MethodResult`` covers every per-method output the harness needs:
- predictive metrics per hospital (AUROC / AUPRC / ECE),
- number of training rounds actually used,
- wall-clock seconds spent in training,
- per-round val-set macro AUC trajectory,
- optional method-specific extras (e.g., final server state, patient AUCs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MethodResult:
    aucs: dict[int, float | None] = field(default_factory=dict)        # AUROC per hospital
    auprcs: dict[int, float | None] = field(default_factory=dict)
    eces: dict[int, float | None] = field(default_factory=dict)
    # Eval-set size behind each hospital's metrics. An AUROC is uninterpretable
    # without them: at 150 windows and ~2 % prevalence there are ~3 positives,
    # which is a 95 % CI of roughly ±0.34.
    ns: dict[int, int] = field(default_factory=dict)                    # windows scored
    n_pos: dict[int, int] = field(default_factory=dict)                 # positives among them
    n_rounds: int = 0                                                   # rounds actually run
    # Which round model selection restored. 0 means the un-adapted population
    # model won on validation, so this method's reported metrics ARE the
    # population baseline's — the cell carries no information about the method
    # and must not be read as agreement between methods. Every ablation knob is
    # a no-op in such a cell, which is what made the task-4 comparison at α=0.1
    # look like three strategies agreeing to 16 decimal places.
    selected_round: int | None = None
    # L2 norm of (selected adapter state − round-0 state), summed over
    # hospitals. Zero exactly when ``selected_round == 0``. Reported so that
    # "the metrics did not move" can be distinguished from "the model did not
    # move": a non-zero norm with identical AUROC means the eval set could not
    # resolve the change, which is a statement about the test set, not the method.
    selected_delta_norm: float | None = None
    # Validation macro AUROC of the SELECTED round, and of round 0. Round 0 is
    # the un-adapted population model, so ``val_auroc_round0`` is
    # population_only's validation score — previously computed for selection and
    # then discarded, which left validation-space comparisons impossible: a
    # method's val number had nothing to be compared against.
    #
    # READ WITH CARE. Because round 0 is a selection candidate,
    # ``val_auroc >= val_auroc_round0`` ALWAYS. So "adaptation beats the
    # population model on validation" is true by construction and carries no
    # information about whether adaptation helps. These two exist to show HOW
    # MUCH selection moved and on how much evidence — not to decide the question.
    val_auroc: float | None = None
    val_auroc_round0: float | None = None
    # A THIRD surface, used for choosing between CONFIGURATIONS.
    #
    # Neither of the other two can do that job. Test cannot: picking the best of
    # N configs on test makes the reported number the maximum of N noisy draws,
    # and with a CI half-width of ±0.0157 against effects of 0.001-0.016 the
    # winner would be indistinguishable from the luckiest draw. Validation cannot
    # either: round 0 is a selection candidate, so val_auroc >= val_auroc_round0
    # always, whatever the config does.
    #
    # ``probe_auroc`` is measured on a slice of the val partition that is never
    # used to select a round. Config decisions read this; the test columns stay
    # untouched until the single final run.
    probe_auroc: float | None = None
    wall_clock_sec: float = 0.0                                         # training-only wall-clock
    val_history: list[tuple[int, float | None]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)                 # method-specific
