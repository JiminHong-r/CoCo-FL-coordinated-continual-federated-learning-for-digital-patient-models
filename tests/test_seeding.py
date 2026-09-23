"""Runs must be reproducible, and replicates must still be possible.

Torch's global RNG was never seeded in the evaluation entry point, so no WP5 cell
could be reproduced: each run drew a different adapter down-projection init, and
because the validation surface is flat enough that round selection is decided by
noise, that alone changed the selected round between identical configurations.

The fix has to keep BOTH properties:

* fixing the config fixes the result (reproducibility), and
* run-to-run variance can still be measured (replication),

which is why there are two seeds. ``seed`` fixes the data — Dirichlet partition,
balanced sampling, loader order. ``run_seed`` fixes training randomness only.
Collapsing them into one knob would make a "seed repeat" silently redraw the
partition, confounding training noise with a different split of patients into
centres.
"""
from __future__ import annotations

import inspect

import torch

from cocofl.adapters.ffn import FFNAdapter


def _init_draw(run_seed: int) -> torch.Tensor:
    """The adapter init that torch's global RNG produces under a given seed."""
    torch.manual_seed(run_seed)
    return FFNAdapter(16, bottleneck=4).down.weight.detach().clone()


def test_same_run_seed_reproduces_the_adapter_init():
    assert torch.equal(_init_draw(42), _init_draw(42))


def test_different_run_seed_changes_it():
    assert not torch.equal(_init_draw(42), _init_draw(43)), (
        "run_seed must actually vary training randomness, or replicates would "
        "be identical and could not measure run-to-run variance"
    )


def test_the_down_projection_is_what_carries_the_randomness():
    """The up-projection is zeroed for identity-at-init, so if the down
    projection were also deterministic the adapter would have no seed
    sensitivity at all and this whole knob would be inert."""
    a = FFNAdapter(16, bottleneck=4)
    assert torch.count_nonzero(a.up.weight) == 0
    assert torch.count_nonzero(a.down.weight) > 0


# --- the entry point actually seeds, and reads both knobs -----------------

def test_run_evaluation_seeds_the_global_torch_rng():
    from cocofl.evaluation import run_evaluation

    src = inspect.getsource(run_evaluation)
    assert "torch.manual_seed(run_seed)" in src, (
        "without this no cell is reproducible"
    )
    assert "cuda.manual_seed_all(run_seed)" in src


def test_run_seed_is_recorded_in_the_results_row():
    """A replicate is only interpretable if the row says which run it was."""
    from cocofl.evaluation import run_evaluation

    assert '"cfg_run_seed"' in inspect.getsource(run_evaluation)
