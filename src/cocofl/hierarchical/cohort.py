"""Cohort-level adapter training (Spec WP4 task 1).

Each clinical centre (hospital) trains its own ``cohort_adapter`` on top of the
frozen population backbone, using its local patient slice (assigned via
Dirichlet partitioning).

Training mode is configurable:
- ``downstream``  : BCE on a task label (mortality / readmission / ICU)
- ``lm``          : next-token CE on the patient timeline

Returns the per-cohort adapter state dicts, which the coordinator then uses
for both downstream patient adaptation and upward propagation into the
population model.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.classifier import TFNDownstreamClassifier
from ..adapters import attach_ffn_adapter, adapter_state_dict


@dataclass
class CohortConfig:
    bottleneck: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    local_epochs: int = 1
    local_steps_per_epoch: int | None = None
    objective: str = "downstream"   # "downstream" | "lm"
    # Catastrophic-forgetting mitigation (Spec WP4 task 4).
    # "distill" : KD against teacher (population for cohort, cohort for patient)
    # "ewc"     : λ/2 · Σ Fisher_i · (θ_i - θ*_i)^2 with θ* = pre-round snapshot
    # "replay"  : alpha · BCE on replay buffer drawn from population
    cl_strategy: str = "none"        # "none" | "distill" | "ewc" | "replay"
    cl_alpha: float = 0.3            # weight of the CL term in the total loss
    cl_temperature: float = 2.0      # softmax temperature for KD
    cl_lambda: float = 100.0         # EWC λ
    cl_replay_capacity: int = 256
    cl_replay_batch_size: int = 8
    cl_fisher_batches: int = 8
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
    #
    # NB: freezing the head is THIS PROJECT'S choice, not a requirement of the
    # task description. The TD uses the word "frozen" exactly once, describing
    # the *population-only baseline* ("frozen population model applied directly
    # without any adaptation"); WP4 task 1 says only "design update rules for
    # centre-specific fine-tuning (e.g., via lightweight adapters or low-rank
    # updates)" — adapters are an example, not a mandate. Earlier comments here
    # claimed the TD specified this; it does not.
    train_head: bool = False
    # Middle ground between a fully frozen head and a fully trained one: unfreeze
    # ONLY the head's bias terms. A class-aware split's heterogeneity is largely
    # label shift (per-centre positive rate 0.42 %-3.10 %), and the correct
    # response to a prior shift is an intercept change. Freezing the intercept
    # forces the adapter to emulate it by distorting features, which is what the
    # measurements show costs ranking while buying calibration. Off by default:
    # enabling it changes the method, not a defect.
    train_head_bias_only: bool = False


def _train_one_cohort_step(classifier: TFNDownstreamClassifier,
                            loader: DataLoader,
                            label_fn: Callable | None,
                            cfg: CohortConfig,
                            device: torch.device,
                            teacher: TFNDownstreamClassifier | None = None,
                            replay_buffer=None,
                            ) -> tuple[int, float]:
    """Run ``cfg.local_epochs`` of local training; return (n_seen, avg_loss).

    If ``teacher`` is provided AND ``cfg.cl_strategy == "distill"``, adds a
    BCE-style knowledge-distillation term against the teacher's soft labels:

        loss = base_loss + cl_alpha * KD(student/T, teacher/T) * T^2

    Used for Spec WP4 task 4 (continual-learning mitigation of catastrophic
    forgetting). For cohort training, the teacher is typically the frozen
    population classifier; for patient training, it is the cohort model.
    """
    classifier.train()
    trainable = [p for p in classifier.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable params — attach cohort adapter first")
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    _pw = (None if cfg.pos_weight is None
           else torch.tensor(float(cfg.pos_weight), device=device))
    use_distill = teacher is not None and cfg.cl_strategy == "distill"
    use_ewc = cfg.cl_strategy == "ewc"
    use_replay = cfg.cl_strategy == "replay" and replay_buffer is not None
    if use_distill:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    T = cfg.cl_temperature

    # EWC setup: snapshot current trainable parameters (θ*) and compute the
    # diagonal Fisher on the LOCAL loader before training starts.
    ewc_theta_star: dict[str, torch.Tensor] | None = None
    ewc_fisher: dict[str, torch.Tensor] | None = None
    if use_ewc:
        from ..personalisation.continual import compute_fisher_tfn
        ewc_theta_star = {n: p.detach().clone() for n, p in classifier.named_parameters()
                          if p.requires_grad}
        if label_fn is not None:
            ewc_fisher = compute_fisher_tfn(
                classifier, loader, label_fn, device, max_batches=cfg.cl_fisher_batches,
            )

    n_seen, total_loss, n_steps = 0, 0.0, 0
    for epoch in range(cfg.local_epochs):
        steps = 0
        for batch in loader:
            if batch is None:
                continue
            x_tuple, y = batch
            x, elapsed, cat, num = (t.to(device) for t in x_tuple[:4])
            if cfg.objective == "downstream":
                if label_fn is None:
                    raise ValueError("downstream objective needs label_fn")
                y_dict = y
                y_t = torch.tensor(label_fn(y_dict), dtype=torch.float32, device=device)
                logits = classifier(
                    input_ids=x, elapsed_times=elapsed,
                    static_categorical=cat, static_numerical=num,
                ).view(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y_t, pos_weight=_pw)
                if use_distill:
                    with torch.no_grad():
                        t_logits = teacher(
                            input_ids=x, elapsed_times=elapsed,
                            static_categorical=cat, static_numerical=num,
                        ).view(-1)
                    t_prob = torch.sigmoid(t_logits / T)
                    kd = F.binary_cross_entropy_with_logits(logits / T, t_prob) * (T * T)
                    loss = loss + cfg.cl_alpha * kd
            elif cfg.objective == "lm":
                y = y.to(device)
                out = classifier.backbone(
                    input_ids=x, elapsed_times=elapsed,
                    static_categorical=cat, static_numerical=num, labels=y,
                )
                loss = out.loss
                if use_distill:
                    with torch.no_grad():
                        t_out = teacher.backbone(
                            input_ids=x, elapsed_times=elapsed,
                            static_categorical=cat, static_numerical=num,
                        )
                    s_logits = out.logits / T
                    t_prob = F.softmax(t_out.logits / T, dim=-1)
                    kd = F.kl_div(F.log_softmax(s_logits, dim=-1), t_prob,
                                  reduction="batchmean") * (T * T)
                    loss = loss + cfg.cl_alpha * kd
            else:
                raise ValueError(f"unknown objective: {cfg.objective!r}")

            # EWC penalty: λ/2 · Σ F_i (θ_i - θ*_i)^2
            if use_ewc and ewc_fisher is not None:
                ewc_pen = torch.zeros((), device=device)
                for n_, p_ in classifier.named_parameters():
                    if n_ not in ewc_fisher or not p_.requires_grad:
                        continue
                    ewc_pen = ewc_pen + (ewc_fisher[n_] * (p_ - ewc_theta_star[n_]).pow(2)).sum()
                loss = loss + 0.5 * cfg.cl_lambda * ewc_pen

            # Replay: BCE on a random batch sampled from the replay buffer
            if use_replay and cfg.objective == "downstream":
                rx, re_, rcat, rnum, ry = replay_buffer.sample(cfg.cl_replay_batch_size)
                rx = rx.to(device); re_ = re_.to(device); rcat = rcat.to(device)
                rnum = rnum.to(device); ry = ry.to(device)
                rlogits = classifier(
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
            n_seen += x.size(0)
            total_loss += float(loss.detach())
            n_steps += 1
            steps += 1
            if cfg.local_steps_per_epoch and steps >= cfg.local_steps_per_epoch:
                break
    return n_seen, total_loss / max(n_steps, 1)


def train_cohort_adapter(
    template: TFNDownstreamClassifier,
    loader: DataLoader,
    cfg: CohortConfig,
    device: torch.device,
    label_fn: Callable | None = None,
) -> tuple[dict[str, torch.Tensor], int, float]:
    """Train one centre's cohort adapter from scratch (template's current state).

    Returns ``(cohort_state, n_seen, avg_loss)`` — state dict contains only the
    ``cohort_adapter.*`` keys (the population backbone is unchanged).
    """
    clf = copy.deepcopy(template)
    attach_ffn_adapter(clf.backbone, bottleneck=cfg.bottleneck, attr="cohort_adapter")
    n_seen, avg_loss = _train_one_cohort_step(clf, loader, label_fn, cfg, device)
    state = adapter_state_dict(clf.backbone, attr="cohort_adapter")
    del clf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state, n_seen, avg_loss


def train_all_cohorts(
    population_clf: TFNDownstreamClassifier,
    hospital_loaders: dict[int, DataLoader],
    cfg: CohortConfig,
    device: torch.device,
    label_fn: Callable | None = None,
) -> dict[int, tuple[dict[str, torch.Tensor], int, float]]:
    """Train an independent cohort adapter for each hospital.

    No cross-hospital aggregation here — this is the per-centre stage. The
    coordinator can later aggregate the deltas back into the population via
    ``upward_prop.propagate_cohorts_to_population``.
    """
    result: dict[int, tuple[dict[str, torch.Tensor], int, float]] = {}
    for hid, loader in hospital_loaders.items():
        state, n_seen, avg_loss = train_cohort_adapter(
            population_clf, loader, cfg, device, label_fn=label_fn,
        )
        result[hid] = (state, n_seen, avg_loss)
    return result
