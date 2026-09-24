"""Patient-level adapter training (WP4 task 2).

Each patient's model starts from their cohort's model. A patient adapter sits on top
of the cohort adapter and trains at a higher learning rate on that patient's data.
Its change (final minus initial) is returned so the coordinator can fold it into the
cohort adapter (WP4 task 3).
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
    # Continual-learning strategy (WP4 task 4). At the patient level the reference
    # (teacher, or theta* for EWC) is the cohort model.
    cl_strategy: str = "none"        # "none" | "distill" | "ewc" | "replay"
    cl_alpha: float = 0.3
    cl_temperature: float = 2.0
    cl_lambda: float = 100.0
    cl_replay_batch_size: int = 8
    cl_fisher_batches: int = 4
    # Positive-class weight for the task BCE. The population head was fitted with a
    # class-weighted loss; training here without one on a centre's ~2 % prevalence
    # pulls predictions back towards the base rate. None = unweighted.
    pos_weight: float | None = None
    # Also train the classifier head? False (default): the head stays frozen and
    # only the patient adapter learns.
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
    # Freeze the cohort adapter; only the patient adapter trains.
    for p in clf.backbone.cohort_adapter.parameters():
        p.requires_grad_(False)
    # Attach the patient adapter (this freezes everything else).
    attach_ffn_adapter(clf.backbone, bottleneck=patient_bottleneck, attr="patient_adapter")
    # The head trains only if this level is set to train it.
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

    Returns ``(personalised_clf, patient_state, patient_delta, avg_loss)``.
    ``patient_delta`` is ``patient_state`` minus the initial state. The adapter
    starts at zero, so the two are equal today, but the difference is computed
    explicitly in case the initialisation changes. ``avg_loss`` is the mean task
    BCE, used for uncertainty weighting.
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

    # Continual-learning teacher for the patient level: the cohort model (population
    # + cohort adapter, no patient adapter).
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

    # EWC: snapshot the patient adapter's starting state (zero) and a small Fisher.
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
    # Mean task BCE over this patient's steps, returned for uncertainty weighting
    # (which uses 1 / loss).
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
        # Count the task loss only. The CL penalties are added below and say nothing
        # about how well this patient is fitted.
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
    # Fourth return value: mean task BCE. Callers that only want the adapter can
    # ignore it with ``*_``.
    avg_loss = bce_sum / max(bce_steps, 1)
    return clf, final, delta, avg_loss


def aggregate_patient_deltas(
    deltas: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    """Weighted average of patient adapter deltas, for the patient-to-cohort step.

    The caller chooses the weights (e.g. by sample count, uncertainty or recency).
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
