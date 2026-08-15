"""Upward propagation: bidirectional adaptation flow (Spec WP4 task 3).

Two directions of upward signal:

  patients (in centre c) ---aggregate-> cohort_c   adapter update
  cohorts (all centres)   ---aggregate-> population adapter update (slow)

Weighting schemes supported (Spec WP4 task 3):
- ``volume``       : weight by sample count
- ``uniform``      : equal weight
- ``recency``      : exponential decay by ELAPSED TIME since the newest update
                     (half-life in days), falling back to position decay with a
                     warning when no timestamps are available
- ``uncertainty``  : inverse loss (lower-loss updates carry more weight)

The scheme is selectable **per level** — the TD asks for a mechanism at
patient->cohort and another at cohort->population, so a single shared knob could
not compare them.

Population update rate is *slower* than cohort/patient — we use an EMA-style
blend ``population_new = (1-η) * population_old + η * aggregated_cohorts``
with small ``η`` (default 0.1) to maintain stability per Spec WP4 task 5.
"""
from __future__ import annotations

from typing import Iterable

import torch


def weighted_state_average(
    states: list[dict[str, torch.Tensor]],
    weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Weighted average of state dicts. ``weights=None`` -> uniform."""
    if not states:
        raise ValueError("no states to average")
    if weights is None:
        weights = [1.0] * len(states)
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("total weight must be positive")
    norm = [w / total for w in weights]
    keys = set(states[0])
    for s in states[1:]:
        if set(s) != keys:
            raise ValueError("states must share keys")
    avg: dict[str, torch.Tensor] = {}
    for k in keys:
        stacked = torch.stack([s[k] for s in states])
        w = torch.tensor(norm, dtype=stacked.dtype, device=stacked.device)
        avg[k] = (stacked * w.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)
    return avg


MICROSECONDS_PER_DAY = 8.64e10


def _weights_for(scheme: str, counts: list[int],
                 losses: list[float] | None = None,
                 times: list[int | None] | None = None,
                 half_life_days: float = 180.0,
                 recency_decay: float = 0.9) -> list[float]:
    """Aggregation weights for one upward-propagation step (Spec WP4 task 3).

    ``recency`` weights by **elapsed time**, not by list position. The TD asks
    for "update recency"; decaying by position in the within-round update list
    does not express that, because every update in a round is equally recent —
    that arm was arbitrary-order decay under a spec-compliant name. Given real
    timestamps, an update ``d`` days older than the newest carries weight
    ``0.5 ** (d / half_life_days)``.

    The position-based decay remains only as a fallback for callers that cannot
    supply timestamps, and it warns, because a silent fallback here reproduces
    exactly the defect this parameter exists to fix.
    """
    if scheme == "volume":
        return [float(c) for c in counts]
    if scheme == "uniform":
        return [1.0] * len(counts)
    if scheme == "recency":
        n = len(counts)
        usable = [t for t in (times or []) if t is not None]
        if times is not None and len(usable) == n and n > 0:
            newest = max(usable)
            hl = max(float(half_life_days), 1e-9) * MICROSECONDS_PER_DAY
            return [0.5 ** ((newest - float(t)) / hl) for t in times]
        import warnings
        warnings.warn(
            "recency weighting fell back to POSITION decay: no usable "
            "timestamps were supplied, so this arm does not weight by update "
            "recency and must not be reported as if it did.",
            stacklevel=2,
        )
        return [recency_decay ** (n - 1 - i) for i in range(n)]
    if scheme == "uncertainty":
        if losses is None:
            raise ValueError("uncertainty weighting requires losses")
        # inverse loss; clamp to avoid div-by-zero
        return [1.0 / max(l, 1e-6) for l in losses]
    raise ValueError(f"unknown weighting scheme: {scheme!r}")


def propagate_patients_to_cohort(
    cohort_state: dict[str, torch.Tensor],
    patient_deltas: list[dict[str, torch.Tensor]],
    patient_counts: list[int],
    eta_cohort: float = 0.3,
    scheme: str = "volume",
    losses: list[float] | None = None,
    times: list[int | None] | None = None,
    half_life_days: float = 180.0,
) -> dict[str, torch.Tensor]:
    """Apply aggregated patient deltas to the cohort adapter via EMA blending.

    ``new_cohort = cohort + eta_cohort * aggregated_delta``

    Note the keys live in the *patient* adapter namespace but are loaded into
    the cohort adapter by key suffix (``down.weight``, ``up.weight``, etc.).
    Both share the same FFNAdapter architecture so the shape match — but the
    caller is responsible for choosing matching bottleneck sizes.
    """
    if not patient_deltas:
        return dict(cohort_state)
    weights = _weights_for(scheme, patient_counts, losses,
                           times=times, half_life_days=half_life_days)
    agg_delta = weighted_state_average(patient_deltas, weights)

    # Strip the namespace ("patient_adapter." -> "cohort_adapter.") for keys.
    # The propagation only works when patient_adapter and cohort_adapter share
    # the same FFNAdapter shape (same bottleneck). Mismatched shapes are skipped
    # with a warning so the caller can debug; the rest of the cohort state is
    # left untouched.
    new_state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for k, v in cohort_state.items():
        if k.startswith("classifier."):
            # Head keys are namespace-free and identical at both levels, so they
            # map straight across. Running them through the adapter-renaming
            # branch would look for "patient_adapter.0.weight" and silently drop
            # every head delta.
            patient_key = k
        else:
            leaf = k.split(".", 1)[1]
            patient_key = f"patient_adapter.{leaf}"
        if patient_key in agg_delta and agg_delta[patient_key].shape == v.shape:
            new_state[k] = v + eta_cohort * agg_delta[patient_key]
        else:
            if patient_key in agg_delta:
                skipped.append(f"{k}: cohort={tuple(v.shape)} vs patient={tuple(agg_delta[patient_key].shape)}")
            new_state[k] = v.clone()
    if skipped:
        import warnings
        warnings.warn(
            "propagate_patients_to_cohort: shape mismatch between patient and cohort "
            f"adapters — propagation skipped for these keys: {skipped}. "
            "Set patient_bottleneck == cohort_bottleneck to enable upward propagation.",
            stacklevel=2,
        )
    return new_state


def propagate_cohorts_to_population(
    population_adapter_state: dict[str, torch.Tensor] | None,
    cohort_states: list[dict[str, torch.Tensor]],
    cohort_counts: list[int],
    eta_population: float = 0.1,
    scheme: str = "volume",
    losses: list[float] | None = None,
    times: list[int | None] | None = None,
    half_life_days: float = 180.0,
) -> dict[str, torch.Tensor]:
    """Slow aggregation of cohort adapters into a population-level adapter.

    If the population had no adapter previously, ``population_adapter_state``
    can be ``None`` -- in which case the aggregated cohort state is scaled by
    ``eta_population`` and returned (starts from zero).

    Strategy: average cohort adapters by ``scheme``, then EMA-blend into
    the population adapter slot. Smaller ``eta_population`` => more stable
    global model (matches Spec WP4 "slow" updates at population level).
    """
    if not cohort_states:
        return dict(population_adapter_state) if population_adapter_state else {}
    weights = _weights_for(scheme, cohort_counts, losses,
                           times=times, half_life_days=half_life_days)
    agg = weighted_state_average(cohort_states, weights)

    if population_adapter_state is None:
        return {k: eta_population * v for k, v in agg.items()}

    # EMA blend; population_adapter_state and agg should share keys (same FFNAdapter shape).
    out: dict[str, torch.Tensor] = {}
    for k, v in population_adapter_state.items():
        if k in agg:
            out[k] = (1.0 - eta_population) * v + eta_population * agg[k]
        else:
            out[k] = v.clone()
    return out
