"""TFN-aware continual learning regularisers for per-patient personalisation.

Parallel to ``cocofl.personalisation.continual.*`` but adapted to TFN batches:
``((x, elapsed, cat, num), y_dict)``.

All three regularisers return an ``extra_loss_fn(clf_fn, x, y, base_loss)``
matching ``loop._train_classifier``'s callback contract. ``clf_fn`` is the
input-only closure produced by ``make_input_only_classifier`` so neither EWC
nor Distill nor Replay needs to know TFN's expanded signature.
"""
from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.classifier import TFNDownstreamClassifier


# --- EWC ---------------------------------------------------------------------

def compute_fisher_tfn(
    classifier: TFNDownstreamClassifier,
    loader: Iterable,
    label_fn: Callable[[dict], list[int]],
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, torch.Tensor]:
    """Empirical diagonal Fisher under BCE loss, for TFN batches."""
    fisher = {
        n: torch.zeros_like(p, device=p.device)
        for n, p in classifier.named_parameters()
        if p.requires_grad
    }
    classifier.eval()
    n_batches = 0
    for i, batch in enumerate(loader):
        if batch is None:
            continue
        x_tuple, y_dict = batch
        x, elapsed, cat, num = (t.to(device, non_blocking=True) for t in x_tuple[:4])
        y = torch.tensor(label_fn(y_dict), dtype=torch.float32, device=device)

        classifier.zero_grad(set_to_none=True)
        logits = classifier(
            input_ids=x, elapsed_times=elapsed,
            static_categorical=cat, static_numerical=num,
        ).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        for n, p in classifier.named_parameters():
            if p.grad is not None and n in fisher:
                fisher[n].add_(p.grad.detach().pow(2))
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    if n_batches == 0:
        raise ValueError("compute_fisher_tfn: no batches consumed")
    for n in fisher:
        fisher[n].div_(n_batches)
    classifier.zero_grad(set_to_none=True)
    return fisher


@dataclass
class EWCRegularizerTFN:
    fisher: dict[str, torch.Tensor]
    theta_star: dict[str, torch.Tensor]
    lambda_ewc: float = 100.0
    _classifier_ref: nn.Module | None = None

    @classmethod
    def from_classifier(cls, classifier: TFNDownstreamClassifier, loader, label_fn,
                        device, lambda_ewc: float = 100.0, max_batches: int | None = None,
                        ) -> "EWCRegularizerTFN":
        fisher = compute_fisher_tfn(classifier, loader, label_fn, device, max_batches)
        theta_star = {n: p.detach().clone() for n, p in classifier.named_parameters()
                      if n in fisher}
        inst = cls(fisher=fisher, theta_star=theta_star, lambda_ewc=lambda_ewc)
        inst._classifier_ref = classifier
        return inst

    def penalty(self, classifier: nn.Module) -> torch.Tensor:
        device = next(classifier.parameters()).device
        total = torch.zeros((), device=device)
        for n, p in classifier.named_parameters():
            if n not in self.fisher:
                continue
            total = total + (self.fisher[n] * (p - self.theta_star[n]).pow(2)).sum()
        return 0.5 * self.lambda_ewc * total

    def as_extra_loss_fn(self):
        """``(clf_fn, x, y, base_loss) -> loss``. We ignore clf_fn since the
        penalty operates on the underlying classifier's parameters directly."""
        clf = self._classifier_ref
        if clf is None:
            raise RuntimeError("EWCRegularizerTFN missing classifier ref")

        def _fn(clf_fn, x, y, base_loss):
            return base_loss + self.penalty(clf)
        return _fn


# --- Distillation ------------------------------------------------------------

def clone_frozen_classifier(classifier: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(classifier)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


@dataclass
class DistillationLossTFN:
    teacher: TFNDownstreamClassifier
    cat: torch.Tensor   # static categorical features for the current personalisation patient
    num: torch.Tensor   # static numerical features
    alpha: float = 1.0
    temperature: float = 1.0

    def __post_init__(self):
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def as_extra_loss_fn(self):
        T = self.temperature

        def _fn(clf_fn, x, y, base_loss):
            elapsed = torch.ones_like(x, dtype=torch.float32)
            with torch.no_grad():
                t_logits = self.teacher(
                    input_ids=x, elapsed_times=elapsed,
                    static_categorical=self.cat, static_numerical=self.num,
                ).view(-1)
            s_logits = clf_fn(x).view(-1)
            t_prob = torch.sigmoid(t_logits / T)
            kd = F.binary_cross_entropy_with_logits(s_logits / T, t_prob) * (T * T)
            return base_loss + self.alpha * kd
        return _fn


# --- Replay ------------------------------------------------------------------

class ReplayBufferTFN:
    """Reservoir of (x, elapsed, cat, num, label) samples drawn from the population
    train loader before personalisation begins. Used to compute a replay-BCE term
    on each personalisation step."""

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.items: deque = deque(maxlen=capacity)

    def fill_from(self, loader, label_fn) -> None:
        """Collect samples until ``capacity`` is reached.

        Note: ETHOS tokenisation produces per-patient *static context* tokens
        whose count varies (DOB → 2 tokens vs single-code → 1 token), so the
        same dataset can yield x tensors of different shapes. The buffer
        commits to the *first* shape it sees and skips any item that doesn't
        match — across multiple ``fill_from`` calls too (e.g., when the
        coordinator fills from each hospital's loader in turn).
        """
        # Persist the target shape across calls so different hospital loaders
        # contribute compatibly-shaped items.
        target_shape = self.items[0][0].shape if self.items else None
        for batch in loader:
            if batch is None:
                continue
            x_tuple, y_dict = batch
            x, elapsed, cat, num = x_tuple[:4]
            labels = label_fn(y_dict)
            for i in range(x.size(0)):
                xi = x[i]
                if target_shape is None:
                    target_shape = xi.shape
                elif xi.shape != target_shape:
                    continue
                self.items.append((
                    xi.detach().clone(),
                    elapsed[i].detach().clone(),
                    cat[i].detach().clone(),
                    num[i].detach().clone(),
                    int(labels[i]),
                ))
                if len(self.items) >= self.capacity:
                    return

    def sample(self, n: int) -> tuple:
        """Returns batched tensors (xs, elapsed, cats, nums, ys)."""
        if not self.items:
            raise ValueError("ReplayBufferTFN: empty buffer")
        n = min(n, len(self.items))
        batch = random.sample(list(self.items), n)
        xs = torch.stack([b[0] for b in batch])
        elapsed = torch.stack([b[1] for b in batch])
        cats = torch.stack([b[2] for b in batch])
        nums = torch.stack([b[3] for b in batch])
        ys = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        return xs, elapsed, cats, nums, ys


def replay_loss_tfn(buffer: ReplayBufferTFN, classifier: TFNDownstreamClassifier,
                    batch_size: int = 8, device: torch.device | None = None,
                    alpha: float = 0.5):
    """Returns an ``extra_loss_fn(clf_fn, x, y, base_loss)`` that adds
    ``alpha * BCE(student(replay), replay_labels)``."""
    def _fn(clf_fn, x, y, base_loss):
        xs, elapsed, cats, nums, ys = buffer.sample(batch_size)
        if device is not None:
            xs, elapsed, cats, nums, ys = (
                xs.to(device), elapsed.to(device), cats.to(device), nums.to(device), ys.to(device),
            )
        # Use the FULL classifier (with patient-specific static features replaced
        # by replay sample's own static features — closer to true population recall).
        logits = classifier(
            input_ids=xs, elapsed_times=elapsed,
            static_categorical=cats, static_numerical=nums,
        ).view(-1)
        replay_bce = F.binary_cross_entropy_with_logits(logits, ys)
        return base_loss + alpha * replay_bce
    return _fn
