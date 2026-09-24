"""Patient-level adapter training (Spec WP4 task 2).

Each patient's model is *initialised from their cohort's model* (not directly
from the population). The patient adapter is attached on top of the cohort
adapter and trained with higher learning rate on patient-specific data.

After patient training completes, we also compute the patient's adapter
*delta* (final - initial) which the coordinator can aggregate into the
cohort's adapter for upward propagation (Spec WP4 task 3).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.classifier import TFNDownstreamClassifier
from ..adapters import (
    attach_ffn_adapter, level_state_dict, load_level_state_dict,
)


@dataclass
class PatientConfig:
    bottleneck: int = 64
    lr: float = 1e-3   # patient updates are FAST — higher than cohort
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    n_iters: int = 100
    # Catastrophic-forgetting mitigation (Spec WP4 task 4). For the patient
    # level the natural reference (teacher / θ*) is the COHORT model.
    cl_strategy: str = "none"        # "none" | "distill" | "ewc" | "replay"
    cl_alpha: float = 0.3
    cl_temperature: float = 2.0
    cl_lambda: float = 100.0
    cl_replay_batch_size: int = 8
    cl_fisher_batches: int = 4
    # Positive-class weight for the task BCE. WP3 fits the population head with
    # a class-weighted loss on a stratified draw; adapting it here with an
    # UNWEIGHTED loss on a centre's natural ~2 % prevalence pulls that head back
    # toward the base rate, so adaptation degrades a well-calibrated model and
    # "no adaptation" wins. None = unweighted (the old behaviour).
    pos_weight: float | None = None
    # Whether adaptation trains the classifier head as well as the adapter.
    # False (default) is the architecture the TD specifies: the population model
    # (backbone + its WP3-fitted head) is frozen and the ADAPTERS carry the
    # adaptation. Retraining a head that was fitted on 20,000 class-balanced
    # reserve windows using ~16,000 windows of one centre's 2 %-prevalence data
    # can only degrade it, which is why adapted methods scored BELOW the frozen
    # population model. Keeping it frozen also restores adapter-only transfer,
    # so the communication-efficiency figure reflects what actually travels.
    train_head: bool = False


def _build_patient_classifier(
    population_clf: TFNDownstreamClassifier,
    cohort_state: dict[str, torch.Tensor],
    cohort_bottleneck: int,
    patient_bottleneck: int,
    train_head: bool = False,
) -> TFNDownstreamClassifier:
    """Clone population + load cohort adapter + attach fresh patient adapter."""
    clf = copy.deepcopy(population_clf)
    attach_ffn_adapter(clf.backbone, bottleneck=cohort_bottleneck, attr="cohort_adapter")
    load_level_state_dict(clf, cohort_state, attr="cohort_adapter")
    # freeze cohort adapter (only patient adapter trains)
    for p in clf.backbone.cohort_adapter.parameters():
        p.requires_grad_(False)
    # attach patient adapter (re-freezes everything but patient adapter)
    attach_ffn_adapter(clf.backbone, bottleneck=patient_bottleneck, attr="patient_adapter")
    # The head trains only when the level is configured to train it; by default
    # the WP3-fitted head is frozen and the patient ADAPTER does the adapting.
    for p in clf.classifier.parameters():
        p.requires_grad_(train_head)
    return clf


def train_patient_adapter(
    population_clf: TFNDownstreamClassifier,
    cohort_state: dict[str, torch.Tensor],
    train_loader: DataLoader,
    label_fn: Callable,
    cfg: PatientConfig,
    device: torch.device,
    cohort_bottleneck: int = 64,
    replay_buffer=None,
) -> tuple[TFNDownstreamClassifier, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Train a patient adapter on top of the given cohort state.

    Returns ``(personalised_clf, patient_state, patient_delta)``.
    ``patient_delta = patient_state - initial_patient_state``; since identity
    initialisation is zero (up.weight=0), ``patient_delta`` equals
    ``patient_state`` after training -- but we compute the difference
    explicitly so the API stays correct if init changes.
    """
    clf = _build_patient_classifier(
        population_clf, cohort_state,
        cohort_bottleneck=cohort_bottleneck,
        patient_bottleneck=cfg.bottleneck,
        train_head=cfg.train_head,
    )
    clf.to(device)
    initial = {k: v.clone() for k, v in level_state_dict(
        clf, attr="patient_adapter", include_head=cfg.train_head).items()}

    _pw = (None if cfg.pos_weight is None
           else torch.tensor(float(cfg.pos_weight), device=device))
    trainable = [p for p in clf.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Continual-learning hooks. Patient-level teacher (for distill) is the
    # cohort-only model: population + cohort adapter without patient adapter.
    teacher = None
    use_distill = cfg.cl_strategy == "distill"
    use_ewc = cfg.cl_strategy == "ewc"
    use_replay = cfg.cl_strategy == "replay" and replay_buffer is not None
    if use_distill:
        teacher = copy.deepcopy(population_clf).to(device)
        attach_ffn_adapter(teacher.backbone, bottleneck=cohort_bottleneck,
                           attr="cohort_adapter")
        load_level_state_dict(teacher, cohort_state, attr="cohort_adapter")
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    T = cfg.cl_temperature

    # EWC: snapshot the patient-adapter init (zero) and a small Fisher.
    ewc_theta_star: dict[str, torch.Tensor] | None = None
    ewc_fisher: dict[str, torch.Tensor] | None = None
    if use_ewc:
        from ..personalisation.continual import compute_fisher_tfn
        ewc_theta_star = {n: p.detach().clone() for n, p in clf.named_parameters()
                          if p.requires_grad}
        ewc_fisher = compute_fisher_tfn(
            clf, train_loader, label_fn, device, max_batches=cfg.cl_fisher_batches,
        )

    clf.train()
    it = iter(train_loader)
    # Mean BCE over this patient's steps. Returned so the coordinator can drive
    # ``weighting_scheme="uncertainty"`` (inverse-loss weights) with a real
    # signal — it previously passed a hardcoded 1.0 for every patient, which
    # made inverse-loss weights uniform and silently turned "uncertainty" into
    # "uniform" under a different name.
    bce_sum, bce_steps = 0.0, 0
    for step in range(cfg.n_iters):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        if batch is None:
            continue
        x_tuple, y_dict = batch
        x, elapsed, cat, num = (t.to(device) for t in x_tuple[:4])
        y = torch.tensor(label_fn(y_dict), dtype=torch.float32, device=device)
        logits = clf(
            input_ids=x, elapsed_times=elapsed,
            static_categorical=cat, static_numerical=num,
        ).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=_pw)
        # Track the TASK loss only — CL penalties (KD/EWC/replay) are added
        # below and are a property of the regulariser, not of how well this
        # patient is fit, which is what uncertainty weighting is asking about.
        bce_sum += float(loss.detach())
        bce_steps += 1
        if teacher is not None:
            with torch.no_grad():
                t_logits = teacher(
                    input_ids=x, elapsed_times=elapsed,
                    static_categorical=cat, static_numerical=num,
                ).view(-1)
            t_prob = torch.sigmoid(t_logits / T)
            kd = F.binary_cross_entropy_with_logits(logits / T, t_prob) * (T * T)
            loss = loss + cfg.cl_alpha * kd
        if use_ewc and ewc_fisher is not None:
            ewc_pen = torch.zeros((), device=device)
            for n_, p_ in clf.named_parameters():
                if n_ not in ewc_fisher or not p_.requires_grad:
                    continue
                ewc_pen = ewc_pen + (ewc_fisher[n_] * (p_ - ewc_theta_star[n_]).pow(2)).sum()
            loss = loss + 0.5 * cfg.cl_lambda * ewc_pen
        if use_replay:
            rx, re_, rcat, rnum, ry = replay_buffer.sample(cfg.cl_replay_batch_size)
            rx = rx.to(device); re_ = re_.to(device); rcat = rcat.to(device)
            rnum = rnum.to(device); ry = ry.to(device)
            rlogits = clf(
                input_ids=rx, elapsed_times=re_,
                static_categorical=rcat, static_numerical=rnum,
            ).view(-1)
            replay_bce = F.binary_cross_entropy_with_logits(rlogits, ry)
            loss = loss + cfg.cl_alpha * replay_bce
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        optimizer.step()

    final = level_state_dict(clf, attr="patient_adapter", include_head=cfg.train_head)
    delta = {k: final[k] - initial[k] for k in final}
    # 4th element: mean task BCE. Callers that only want the adapter can keep
    # unpacking three values via ``*_``; the coordinator needs this for
    # uncertainty weighting.
    avg_loss = bce_sum / max(bce_steps, 1)
    return clf, final, delta, avg_loss


def aggregate_patient_deltas(
    deltas: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    """Weighted average of patient adapter deltas — used for upward propagation
    (patient -> cohort). Caller decides the weighting scheme (e.g., by sample
    count, uncertainty, or recency).
    """
    if not deltas:
        raise ValueError("no deltas to aggregate")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("total weight must be positive")
    norm = [w / total for w in weights]
    keys = set(deltas[0])
    for d in deltas[1:]:
        if set(d) != keys:
            raise ValueError("deltas must share keys (compatible adapter shapes)")
    agg: dict[str, torch.Tensor] = {}
    for k in keys:
        stacked = torch.stack([d[k] for d in deltas])
        w = torch.tensor(norm, dtype=stacked.dtype, device=stacked.device)
        agg[k] = (stacked * w.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)
    return agg
