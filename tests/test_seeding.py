"""Runs must be reproducible, and replicates must still be possible.

Two seeds keep both properties. ``seed`` fixes the data (Dirichlet partition,
balanced sampling, loader order). ``run_seed`` fixes only training randomness
(adapter initialisation). With a single seed, a "repeat with another seed" would
also redraw which patients belong to which centre. Before seeding, identical
configurations could select different rounds purely from the adapter initialisation.
"""
from __future__ import annotations

import inspect

import torch

from cocofl.adapters.ffn import FFNAdapter


def _init_draw(run_seed: int) -> torch.Tensor:
    """The adapter initialisation torch's global RNG produces under a given seed."""
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
    """The up-projection starts at zero, so the down-projection is where the randomness is; if it were fixed, the seed would change nothing."""
    a = FFNAdapter(16, bottleneck=4)
    assert torch.count_nonzero(a.up.weight) == 0
    assert torch.count_nonzero(a.down.weight) > 0


# The entry point seeds, and reads both settings

def test_run_evaluation_seeds_the_global_torch_rng():
    from cocofl.evaluation import run_evaluation

    src = inspect.getsource(run_evaluation)
    assert "torch.manual_seed(run_seed)" in src, (
        "without this no cell is reproducible"
    )
    assert "cuda.manual_seed_all(run_seed)" in src


def test_run_seed_is_recorded_in_the_results_row():
    """A replicate only makes sense if its row says which run it was."""
    from cocofl.evaluation import run_evaluation

    assert '"cfg_run_seed"' in inspect.getsource(run_evaluation)
