"""Every keyword the harness passes must exist on the callee.

A real crash this would have caught: ``probe_loaders`` was added to
``_round_loop_with_early_stopping`` and ``run_standard_fedavg``, and passed from
``run_evaluation.main`` to all four training methods — but never added to
``run_local_finetune`` or ``run_top_down_only``. Every unit test passed, because
they inspected SOURCE TEXT for the presence of strings rather than checking that
the call contract holds. The failure surfaced only in an end-to-end smoke run, as
``TypeError: run_local_finetune() got an unexpected keyword argument``, which
would have killed the next full 5-method cell on launch.

Static, no GPU, no data: parse the call sites in ``main`` and check every keyword
against the callee's signature.
"""
from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from cocofl.evaluation import baselines, run_evaluation

TARGETS = {
    "run_local_finetune": baselines.run_local_finetune,
    "run_top_down_only": baselines.run_top_down_only,
    "run_standard_fedavg": baselines.run_standard_fedavg,
    "eval_population_only": baselines.eval_population_only,
    "run_hierarchical_bidir": run_evaluation.run_hierarchical_bidir,
    "evaluate_patient_layer": baselines.evaluate_patient_layer,
}


def _call_sites():
    """Every call to a TARGETS function anywhere in run_evaluation.py."""
    src = pathlib.Path(inspect.getfile(run_evaluation)).read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in TARGETS:
                yield node.func.id, node


def test_the_harness_actually_calls_these():
    found = {name for name, _ in _call_sites()}
    assert "run_local_finetune" in found and "run_hierarchical_bidir" in found, found


@pytest.mark.parametrize("name", sorted(TARGETS))
def test_every_keyword_at_every_call_site_exists_on_the_callee(name):
    sig = inspect.signature(TARGETS[name])
    accepts_kwargs = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    for site_name, node in _call_sites():
        if site_name != name:
            continue
        for kw in node.keywords:
            if kw.arg is None:      # **kwargs splat at the call site
                continue
            assert accepts_kwargs or kw.arg in sig.parameters, (
                f"run_evaluation passes {name}({kw.arg}=...) but {name} has no "
                f"such parameter — this is a TypeError at runtime, not a warning"
            )


@pytest.mark.parametrize("name", sorted(TARGETS))
def test_no_call_site_omits_a_required_parameter(name):
    """Positional-or-keyword params with no default must be supplied somehow."""
    sig = inspect.signature(TARGETS[name])
    required = [n for n, p in sig.parameters.items()
                if p.default is p.empty
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    for site_name, node in _call_sites():
        if site_name != name:
            continue
        supplied = {kw.arg for kw in node.keywords if kw.arg}
        n_positional = len(node.args)
        unmet = [r for i, r in enumerate(required)
                 if i >= n_positional and r not in supplied]
        assert not unmet, f"{name} call is missing required parameter(s): {unmet}"


def test_probe_loaders_specifically_reaches_all_four_methods():
    """The regression that motivated this file."""
    for fn in ("run_local_finetune", "run_top_down_only",
               "run_standard_fedavg", "run_hierarchical_bidir"):
        assert "probe_loaders" in inspect.signature(TARGETS[fn]).parameters, fn
