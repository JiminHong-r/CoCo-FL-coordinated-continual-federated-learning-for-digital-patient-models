"""WP5 metrics: personalisation gain, retention, BWT, FWT.

All AUC inputs may be ``None`` (eval set produced only one class, etc.) — the
metric helpers return ``None`` in that case rather than raising.
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
    """Fraction of the pre-adaptation performance that survives:
    ``retention = after / before``. ``=1.0`` means perfect preservation."""
    return _safe_div(after, before)


def bwt(round_aucs: list[float | None]) -> float | None:
    """Backward Transfer: mean AUC change between the FIRST measurement
    (after the model was first adapted) and the LAST measurement (after all
    subsequent adaptation rounds).

    BWT > 0 : continual adaptation IMPROVED early-task performance
    BWT = 0 : neutral
    BWT < 0 : catastrophic forgetting

    SCOPE, because the TD's wording differs. The TD asks for "performance change
    on past TASKS after adaptation". A WP5 cell trains one task, so there are no
    past tasks to revisit — this measures drift on the SAME task's trajectory set
    between the first and last round. That is the closest available reading, and
    it is what the reported BWT means; it is not cross-task backward transfer and
    must not be described as such.

    Skips None entries.
    """
    vals = [v for v in round_aucs if v is not None]
    if len(vals) < 2:
        return None
    return float(vals[-1] - vals[0])


def fwt(rec: AucPerLevel, baseline_population: float | None) -> float | None:
    """Forward Transfer: how much the *cohort/patient* adapted model
    helps on a held-out patient population (compared to the un-adapted
    population baseline).

    Approximation here: ``cohort_auc - population_baseline_auc``. A more
    rigorous FWT would evaluate on a brand-new task or patient unseen during
    cohort training; for our setting (per-task BCE) we use the population
    baseline as the reference.
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

        SE = sqrt( [A(1-A) + (n_p-1)(Q1-A²) + (n_n-1)(Q2-A²)] / (n_p·n_n) )
        Q1 = A/(2-A),  Q2 = 2A²/(1+A)

    Reported alongside every AUROC so the eval-set size is visible in the
    numbers themselves. It is what makes the small-sample regime concrete: at
    n=150 and ~2 % prevalence (≈3 positives) this returns ≈0.34, which is the
    same order as the entire spread between methods — i.e. per-hospital
    mortality AUROCs at that size carry almost no information on their own.

    Note this is the CI for a *single* AUROC. Method-to-method comparisons in
    this harness are paired (identical eval windows), so their differences are
    estimated more precisely than these independent intervals suggest.
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
    """Number of parameters transmitted for one level's update.

    Spec WP5 axis 6 asks for the "number of parameters updated per
    synchronisation round (per level)" and for total communication cost "in
    equivalent parameter counts" — the harness reported bytes only, which is the
    same information but not the unit the TD asks for.
    """
    if state is None:
        return None
    return sum(int(v.numel()) for v in state.values())


def naive_full_model_parameters(model) -> int:
    """Parameter count of the FULL model — the naïve-federation reference."""
    return sum(int(p.numel()) for p in model.parameters())


def naive_full_model_bytes(model, dtype_bytes: int = 4) -> int:
    """Bytes to transmit the FULL model state (backbone + classifier head).

    This is the cost of *naïve* federated learning where each centre sends
    the entire model state every round (no adapters, no parameter-efficient
    methods). Used as the comparison baseline for Spec WP5 axis 6.
    """
    return sum(int(p.numel()) for p in model.parameters()) * dtype_bytes


def total_communication_cost(per_round_bytes: int, num_hospitals: int,
                              num_rounds: int) -> int:
    """Round-trip cost over the full training run.

    Each round: ``num_hospitals`` uploads + 1 server broadcast = ``num_hospitals + 1``
    transfers of size ``per_round_bytes``. ``num_rounds`` rounds total.
    """
    return per_round_bytes * (num_hospitals + 1) * num_rounds


def comm_savings_ratio(method_bytes_per_round: int, naive_bytes_per_round: int
                        ) -> float | None:
    """Compression factor: how many times *cheaper* the method is than naïve FL.

    A value of 10 means the method transmits 10× less per round.
    Returns ``None`` when ``method_bytes_per_round`` is zero.
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
