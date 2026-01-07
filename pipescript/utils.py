from __future__ import annotations

import ast
from typing import Any

import pyccolo as pyc


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
