"""WP5 metrics: personalisation gain, retention, BWT, FWT, communication cost.

AUROC inputs may be ``None`` (e.g. a single-class eval set); the helpers then return
``None`` instead of raising.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class AucPerLevel:
    """Per-(task, hospital) AUC at each adaptation level."""
    task: str
    hospital: int
    population: float | None = None
    cohort: float | None = None
    patient: float | None = None  # pooled across that hospital's patients


def _safe_sub(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return float(num) / float(den)


def delta_cohort(rec: AucPerLevel) -> float | None:
    """Personalisation gain at cohort level: cohort - population."""
    return _safe_sub(rec.cohort, rec.population)


def delta_patient(rec: AucPerLevel) -> float | None:
    """Personalisation gain at patient level: patient - cohort."""
    return _safe_sub(rec.patient, rec.cohort)


def retention(after: float | None, before: float | None) -> float | None:
    """Share of the starting performance that remains: ``after / before`` (1.0 = fully kept)."""
    return _safe_div(after, before)


def bwt(round_aucs: list[float | None]) -> float | None:
    """Backward transfer: mean AUROC change from the first round to the last.

    BWT > 0: later rounds improved on the first adapted model; BWT < 0: forgetting.

    The TD asks for performance change on past tasks, but one WP5 cell trains one
    task, so this measures drift on the same task between the first and last round.
    It is not cross-task backward transfer. None entries are skipped.
    """
    vals = [v for v in round_aucs if v is not None]
    if len(vals) < 2:
        return None
    return float(vals[-1] - vals[0])


def fwt(rec: AucPerLevel, baseline_population: float | None) -> float | None:
    """Same-centre transfer: ``cohort_auc - population_baseline_auc``.

    Kept for comparison only (``fwt_hier_samecentre``). Scored on the centre's own
    data it is just Delta_cohort rescaled. The reported forward transfer
    (``fwt_hier_global``) scores each cohort model on a reference set pooled from
    all centres; see ``run_evaluation``.
    """
    if rec.cohort is None or baseline_population is None:
        return None
    return float(rec.cohort) - float(baseline_population)


def macro_average(values: Iterable[float | None]) -> float | None:
    """Macro-average over non-None entries."""
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def variance(values: Iterable[float | None]) -> float | None:
    """Sample variance over non-None entries. Returns None if < 2 entries."""
    xs = [float(v) for v in values if v is not None]
    if len(xs) < 2:
        return None
    mu = sum(xs) / len(xs)
    return float(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def auroc_ci95_halfwidth(auc: float | None, n: int | None, n_pos: int | None
                          ) -> float | None:
    """Half-width of the 95 % CI for an AUROC (Hanley & McNeil, 1982).

    ::

        SE = sqrt( [A(1-A) + (n_p-1)(Q1-A²) + (n_n-1)(Q2-A²)] / (n_p·n_n) )
        Q1 = A/(2-A),  Q2 = 2A²/(1+A)

    Reported next to every AUROC so its precision is visible. At 150 windows and ~2
    % prevalence (about 3 positives) it is about 0.34, larger than the whole spread
    between methods.

    This is the interval for a single AUROC. Method comparisons here are paired
    (same windows), so their differences are more precise than these intervals
    suggest.
    """
    if auc is None or not n_pos or not n:
        return None
    n_neg = n - n_pos
    if n_pos < 1 or n_neg < 1:
        return None
    a = float(auc)
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (
        a * (1.0 - a)
        + (n_pos - 1) * (q1 - a * a)
        + (n_neg - 1) * (q2 - a * a)
    ) / (n_pos * n_neg)
    if var <= 0:
        return 0.0
    return float(1.96 * var ** 0.5)


def communication_bytes(state: dict, dtype_bytes: int = 4) -> int:
    """Total bytes to transmit a state-dict (assuming float32 by default)."""
    return sum(int(v.numel()) for v in state.values()) * dtype_bytes


def parameters_updated(state: dict | None) -> int | None:
    """Number of parameters sent for one level's update.

    WP5 axis 6 asks for parameters updated per synchronisation round, per level, and
    for total cost in parameter counts, not only bytes.
    """
    if state is None:
        return None
    return sum(int(v.numel()) for v in state.values())


def naive_full_model_parameters(model) -> int:
    """Parameter count of the full model, the reference for naive federation."""
    return sum(int(p.numel()) for p in model.parameters())


def naive_full_model_bytes(model, dtype_bytes: int = 4) -> int:
    """Bytes to send the full model state (backbone + classifier head).

    The cost of naive federated learning, where every centre sends the whole model
    every round. The comparison point for WP5 axis 6.
    """
    return sum(int(p.numel()) for p in model.parameters()) * dtype_bytes


def total_communication_cost(per_round_bytes: int, num_hospitals: int,
                              num_rounds: int) -> int:
    """Total transfer cost over the whole run.

    Each round has ``num_hospitals`` uploads plus one broadcast, i.e.
    ``num_hospitals + 1`` transfers of ``per_round_bytes``, over ``num_rounds``
    rounds.
    """
    return per_round_bytes * (num_hospitals + 1) * num_rounds


def comm_savings_ratio(method_bytes_per_round: int, naive_bytes_per_round: int
                        ) -> float | None:
    """How many times cheaper the method is than naive FL per round.

    10 means ten times less data per round. ``None`` when ``method_bytes_per_round``
    is zero.
    """
    if method_bytes_per_round <= 0:
        return None
    return float(naive_bytes_per_round) / float(method_bytes_per_round)


def auprc(preds: list[float], labels: list[int]) -> float | None:
    """Area Under the Precision-Recall Curve. ``None`` when labels are single-class."""
    if not preds or len(set(labels)) < 2:
        return None
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(labels, preds))


def expected_calibration_error(preds: list[float], labels: list[int], n_bins: int = 10
                                ) -> float | None:
    """Expected Calibration Error (ECE) over equal-width probability bins.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Lower is better; 0 means perfect calibration.
    """
    if not preds:
        return None
    import numpy as np
    p = np.asarray(preds, dtype=float)
    y = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(p)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        bin_conf = p[mask].mean()
        bin_acc = y[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)
