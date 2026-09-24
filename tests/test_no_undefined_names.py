"""No function may use a name it never defines.

A patch once added a block referencing ``coord`` to ``run_standard_fedavg``, which
has no coordinator. It compiled, imported and passed every unit test, then crashed
with ``NameError`` in an end-to-end run. (An unguarded ``str.replace`` had inserted
the block in two functions.)

This walks each function, collects every name it binds (arguments, assignments,
imports, comprehension targets, except handlers, walrus) and flags any other name it
reads that is not module-level or builtin.
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
            # A nested function adds its own name and its parameters to what is in
            # scope inside it; otherwise a helper's own parameter would be reported
            # as undefined.
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
    """The specific regression: plain FedAvg has no coordinator."""
    from cocofl.evaluation import baselines

    src = inspect.getsource(baselines.run_standard_fedavg)
    assert "coord." not in src, (
        "run_standard_fedavg must not use a coordinator — it does its own "
        "weighted average and broadcast, with no hierarchy"
    )
