"""No function may reference a name it never binds.

This exists because of a real crash that 232 tests missed. A patch inserted a
probe-evaluation block into ``run_standard_fedavg`` that referenced ``coord`` —
a coordinator that function does not have, because flat FedAvg does not use one.
It compiled, it imported, every unit test passed, and it died only when an
end-to-end run reached that method:

    NameError: name 'coord' is not defined   (baselines.py:544)

The cause was an unguarded ``str.replace``, which replaces EVERY occurrence: the
target text appeared in both ``_round_loop_with_early_stopping`` and
``run_standard_fedavg``, so the coordinator-based block landed in both.

Signature tests cannot catch this — the signature was fine, the body was not.
This walks each function, collects everything it binds (arguments, assignments,
imports, comprehension targets, except-handlers, walrus) and flags loads of
anything else that is not a module-level or builtin name.
"""
from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

MODULES = [
    "cocofl.evaluation.baselines",
    "cocofl.evaluation.run_evaluation",
    "cocofl.evaluation.metrics",
    "cocofl.hierarchical.coordinator",
    "cocofl.hierarchical.upward_prop",
    "cocofl.adapters.ffn",
    "cocofl.data.elapsed",
]


def _module_level_names(tree: ast.Module) -> set[str]:
    import builtins
    names = set(dir(builtins)) | {"__name__", "__file__", "__doc__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _args_of(fn) -> set[str]:
    a = fn.args
    out = {x.arg for x in a.args} | {x.arg for x in a.kwonlyargs} | {x.arg for x in a.posonlyargs}
    if a.vararg:
        out.add(a.vararg.arg)
    if a.kwarg:
        out.add(a.kwarg.arg)
    return out


def _bound_in(fn: ast.FunctionDef) -> set[str]:
    bound = _args_of(fn)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.comprehension) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # A NESTED function contributes its own name AND its parameters to
            # what is legitimately in scope inside this subtree — otherwise a
            # helper like ``def _cap_str(n)`` reports its own parameter as
            # undefined.
            bound.add(node.name)
            bound |= _args_of(node)
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.Lambda):
            bound |= _args_of(node)
    return bound


@pytest.mark.parametrize("modname", MODULES)
def test_no_function_references_an_undefined_name(modname):
    mod = __import__(modname, fromlist=["_"])
    src = pathlib.Path(inspect.getfile(mod)).read_text()
    tree = ast.parse(src)
    module_names = _module_level_names(tree)
    offenders = []
    top_level = [n for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        top_level += [n for n in cls.body
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for fn in top_level:
        bound = _bound_in(fn)
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in bound and node.id not in module_names:
                    offenders.append(f"{fn.name}:{node.lineno} -> '{node.id}'")
    assert not offenders, (
        f"{modname} references names that are never bound in scope. This is a "
        f"runtime NameError waiting for the code path to be exercised:\n  "
        + "\n  ".join(sorted(set(offenders)))
    )


def test_fedavg_does_not_reference_a_coordinator():
    """The specific regression: flat FedAvg has no coordinator."""
    from cocofl.evaluation import baselines

    src = inspect.getsource(baselines.run_standard_fedavg)
    assert "coord." not in src, (
        "run_standard_fedavg must not use a coordinator — it does its own "
        "weighted average and broadcast, with no hierarchy"
    )
