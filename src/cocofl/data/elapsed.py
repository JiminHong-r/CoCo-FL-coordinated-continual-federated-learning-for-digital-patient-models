"""Real inter-event times for an inference window (WP3 task 1).

WHY THIS EXISTS
---------------
``TFNFullModel`` is a T-LSTM whose cell discounts the short-term memory component
by ``T = 1/log(Δt + e)`` — the "time-decay mechanism for irregular intervals" the
TD asks for in WP3 task 1. It only does anything if real Δt reaches it.

The population backbone WAS trained with real times (``train/run_training.py``
uses ``collate_with_static``, which computes ``diff(times)`` in hours). But every
downstream path — the WP3 probe in ``downstream/finetune.py`` and the whole
WP4/WP5 harness via ``run_ablation.make_collate`` — synthesised
``elapsed = torch.ones_like(xs)``. So T collapsed to the constant 0.7615 and the
mechanism was inert in every reported number, while the backbone was run outside
the input distribution it was trained on. Measured over 818,800 real gaps:

    real Δt     T at training      T at eval (Δt=1)
    p50-p90 = 0 h      1.0000            0.7615     <- over-discounted
    p99 = 120 h        0.2079            0.7615     <- under-discounted

Wrong in both directions: ~90 % of transitions are same-timestamp (ETHOS emits
several tokens per event) and were being discounted when they should not be,
while the clinically meaningful multi-day gaps were barely discounted at all.

RECONSTRUCTION IS EXACT, NOT INFERRED
-------------------------------------
``InferenceDataset.__getitem__`` builds ``x = cat(patient_context, timeline)``
where ``timeline = tokens[data_start_idx : data_idx + 1]`` and ``data_start_idx``
is the patient's offset, capped to ``timeline_size``. The task datasets already
record ``data_idx`` in their label dict, so the same window can be recomputed
with the same formula rather than approximated — which matters, because silently
MISALIGNED times would be worse than constant ones.
"""
from __future__ import annotations

import torch

MICROSECONDS_PER_HOUR = 3_600_000_000.0


def window_token_range(dataset, data_idx: int) -> tuple[int, int]:
    """The ``[start, end]`` token range ``InferenceDataset`` used for this sample.

    Mirrors ``InferenceDataset.__getitem__`` exactly: start at the patient's
    offset, then clip to at most ``timeline_size`` tokens ending at ``data_idx``.
    """
    end = int(data_idx)
    start = int(dataset.patient_offset_at_idx[end])
    if end - start + 1 > dataset.timeline_size:
        start = end + 1 - dataset.timeline_size
    return start, end


def elapsed_for_window(dataset, data_idx: int, total_len: int) -> torch.Tensor:
    """Per-position elapsed hours for one window, aligned to ``x``.

    Patient-context tokens are atemporal and get 1.0 (log-stable, matching
    ``collate_with_static``). The first timeline token has no predecessor inside
    the window, so it gets 0.0.
    """
    start, end = window_token_range(dataset, data_idx)
    times = torch.as_tensor(dataset.times[start : end + 1], dtype=torch.float64)
    if times.numel() >= 2:
        deltas = torch.diff(times).clamp(min=0) / MICROSECONDS_PER_HOUR
        timeline = torch.cat([torch.zeros(1, dtype=torch.float64), deltas])
    else:
        timeline = torch.zeros(max(times.numel(), 0), dtype=torch.float64)

    n_context = total_len - timeline.numel()
    if n_context < 0:                      # window longer than x: truncate the head
        timeline = timeline[-total_len:]
        n_context = 0
    out = torch.cat([torch.ones(n_context, dtype=torch.float64), timeline])
    if out.numel() < total_len:            # pad defensively; must never happen
        out = torch.cat([out, torch.ones(total_len - out.numel(), dtype=torch.float64)])
    return out[:total_len].to(torch.float32)


def dataset_supports_elapsed(dataset) -> bool:
    """Whether the arrays needed to reconstruct real times are present."""
    return all(getattr(dataset, a, None) is not None
               for a in ("times", "patient_offset_at_idx", "timeline_size"))
