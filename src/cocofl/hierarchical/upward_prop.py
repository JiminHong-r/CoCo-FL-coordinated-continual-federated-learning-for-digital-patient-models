"""Upward propagation (WP4 task 3).

Two upward steps::

    patients (in centre c) --aggregate--> cohort_c adapter
    cohorts (all centres)  --aggregate--> population adapter (slow)

Weighting schemes, selectable separately for each step:

- ``volume``: by sample count - ``uniform``: equal weights - ``recency``:
exponential decay by time since the newest update (half-life in days); falls back to
decay by list position, with a warning, when there are no timestamps -
``uncertainty``: inverse loss (lower loss counts more)

The population moves slowly: ``population_new = (1 - eta) * population_old + eta *
aggregated_cohorts`` with a small ``eta`` (default 0.1), for stability (WP4 task 5).
"""
from __future__ import annotations


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
    """Aggregation weights for one upward step (WP4 task 3).

    ``recency`` decays by elapsed time: an update ``d`` days older than the newest
    gets weight ``0.5 ** (d / half_life_days)``. Decaying by list position would not
    mean recency, since every update within a round is equally recent. Position
    decay remains only as a fallback when there are no timestamps, and it warns.
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
        # Inverse loss, clamped to avoid dividing by zero.
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
    """Add the aggregated patient deltas to the cohort adapter.

    ``new_cohort = cohort + eta_cohort * aggregated_delta``

    The deltas use ``patient_adapter.*`` keys and are matched to the cohort adapter
    by key suffix (``down.weight``, ``up.weight``, ...). Both adapters must have the
    same bottleneck size.
    """
    if not patient_deltas:
        return dict(cohort_state)
    weights = _weights_for(scheme, patient_counts, losses,
                           times=times, half_life_days=half_life_days)
    agg_delta = weighted_state_average(patient_deltas, weights)

    # Map "patient_adapter." keys to "cohort_adapter.". Keys whose shapes do not
    # match (different bottleneck) are skipped with a warning, leaving the rest of
    # the cohort state as it was.
    new_state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for k, v in cohort_state.items():
        if k.startswith("classifier."):
            # Head keys have no adapter prefix and are the same at both levels, so
            # they map straight across; renaming them would drop every head delta.
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
    """Slowly fold the cohort adapters into the population adapter.

    The cohorts are averaged by ``scheme`` and then blended into the population
    adapter. If there is no population adapter yet (``None``), the average times
    ``eta_population`` is returned, i.e. the blend starts from zero. A smaller
    ``eta_population`` gives a more stable population model (WP4 "slow" updates).
    """
    if not cohort_states:
        return dict(population_adapter_state) if population_adapter_state else {}
    weights = _weights_for(scheme, cohort_counts, losses,
                           times=times, half_life_days=half_life_days)
    agg = weighted_state_average(cohort_states, weights)

    if population_adapter_state is None:
        return {k: eta_population * v for k, v in agg.items()}

    # Blend; both states have the same keys (same adapter shape).
    out: dict[str, torch.Tensor] = {}
    for k, v in population_adapter_state.items():
        if k in agg:
            out[k] = (1.0 - eta_population) * v + eta_population * agg[k]
        else:
            out[k] = v.clone()
    return out
