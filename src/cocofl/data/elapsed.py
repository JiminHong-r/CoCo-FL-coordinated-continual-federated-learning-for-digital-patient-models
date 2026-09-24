"""Real inter-event times for an inference window (WP3 task 1).

The T-LSTM shrinks its short-term memory by ``T = 1/log(dt + e)``, the time decay
for irregular intervals that WP3 task 1 asks for. It only works if real time gaps
reach it. The backbone is trained with real gaps (``diff(times)`` in hours), so
evaluation must supply them too. With a constant gap of 1, T is stuck at 0.7615,
which is wrong both ways (measured over 818,800 real gaps)::

    real dt          T with real dt   T with dt = 1
    p50-p90 = 0 h    1.0000           0.7615    (too much decay)
    p99 = 120 h      0.2079           0.7615    (too little decay)

About 90 % of gaps are zero, because ETHOS emits several tokens per event.

The gaps are rebuilt exactly, not estimated. ``InferenceDataset.__getitem__`` builds
``x = cat(patient_context, timeline)`` with ``timeline = tokens[data_start_idx :
data_idx + 1]``, where ``data_start_idx`` is the patient's offset, capped to
``timeline_size``. The task datasets record ``data_idx``, so the same window can be
recomputed with the same formula. Misaligned times would be worse than constant
ones.
"""
from __future__ import annotations

import torch

MICROSECONDS_PER_HOUR = 3_600_000_000.0


def window_token_range(dataset, data_idx: int) -> tuple[int, int]:
    """The ``[start, end]`` token range ``InferenceDataset`` used for this sample.

    Same rule as ``InferenceDataset.__getitem__``: start at the patient's offset,
    then keep at most ``timeline_size`` tokens ending at ``data_idx``.
    """
    end = int(data_idx)
    start = int(dataset.patient_offset_at_idx[end])
    if end - start + 1 > dataset.timeline_size:
        start = end + 1 - dataset.timeline_size
    return start, end


def elapsed_for_window(dataset, data_idx: int, total_len: int) -> torch.Tensor:
    """Elapsed hours for each position of one window, aligned with ``x``.

    Patient-context tokens have no time and get 1.0 (as in ``collate_with_static``).
    The first timeline token has no earlier token in the window, so it gets 0.0.
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
    """Whether the dataset has the arrays needed to rebuild real times."""
    return all(getattr(dataset, a, None) is not None
               for a in ("times", "patient_offset_at_idx", "timeline_size"))
