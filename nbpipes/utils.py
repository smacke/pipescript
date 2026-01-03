from __future__ import annotations

import ast
from typing import Any

import pyccolo as pyc

from nbpipes.api import allow_pipelines_in_loops_and_calls


# for unittest.mock patching
def _get_user_ns_impl() -> dict[str, Any] | None:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None:
            return shell.user_ns
    except ImportError:
        pass
    return None


def get_user_ns() -> dict[str, Any] | None:
    return _get_user_ns_impl()


def has_augmentations(
    node_or_id: ast.AST | list[ast.AST] | int,
    expected_augs: pyc.AugmentationSpec | set[pyc.AugmentationSpec] | None = None,
) -> bool:
    if isinstance(node_or_id, list):
        return any(
            has_augmentations(field, expected_augs=expected_augs)
            for field in node_or_id
        )
    node_id = node_or_id if isinstance(node_or_id, int) else id(node_or_id)
    actual_augs = pyc.BaseTracer.get_augmentations(node_id)
    if expected_augs is None:
        return bool(actual_augs)
    elif not isinstance(expected_augs, set):
        expected_augs = {expected_augs}
    return bool(actual_augs & expected_augs)


def is_outer_or_allowlisted(node_or_id: ast.AST | int) -> bool:
    node_id = node_or_id if isinstance(node_or_id, int) else id(node_or_id)
    if pyc.is_outer_stmt(node_id):
        return True
    containing_stmt = pyc.BaseTracer.containing_stmt_by_id.get(node_id)
    parent_stmt = pyc.BaseTracer.parent_stmt_by_id.get(
        node_id if containing_stmt is None else id(containing_stmt)
    )
    while parent_stmt is not None:
        if isinstance(parent_stmt, ast.With):
            context_expr = parent_stmt.items[0].context_expr
            if (
                isinstance(context_expr, ast.Call)
                and isinstance(context_expr.func, ast.Name)
                and context_expr.func.id == allow_pipelines_in_loops_and_calls.__name__
            ):
                return True
        elif isinstance(parent_stmt, (ast.AsyncFunctionDef, ast.FunctionDef)):
            for deco in parent_stmt.decorator_list:
                if isinstance(deco, ast.Name):
                    actual_deco = deco
                elif isinstance(deco, ast.Call) and isinstance(deco.func, ast.Name):
                    actual_deco = deco.func
                else:
                    continue
                if actual_deco.id == allow_pipelines_in_loops_and_calls.__name__:
                    return True
        parent_stmt = pyc.BaseTracer.parent_stmt_by_id.get(id(parent_stmt))
    return False
