from __future__ import annotations

import ast
import weakref

from pyccolo.utils import clone_function

try:
    from executing.executing import find_node_ipython as orig_find_node_ipython

    orig_find_node_ipython_cloned = clone_function(orig_find_node_ipython)  # type: ignore[arg-type]
except ImportError:
    orig_find_node_ipython = None  # type: ignore[assignment]
    orig_find_node_ipython_cloned = None  # type: ignore[assignment]


frame_to_node_mapping: weakref.WeakValueDictionary[tuple[str, int], ast.AST] = (
    weakref.WeakValueDictionary()
)


def find_node_ipython(frame, last_i, stmts, source):
    decorator, node = orig_find_node_ipython_cloned(frame, last_i, stmts, source)
    if decorator is None and node is None:
        return None, frame_to_node_mapping.get(
            (frame.f_code.co_filename, frame.f_lineno)
        )
    else:
        return decorator, node


def patch_find_node_ipython():
    if orig_find_node_ipython is None or orig_find_node_ipython_cloned is None:
        return
    orig_find_node_ipython.__code__ = find_node_ipython.__code__
    orig_find_node_ipython.__globals__["orig_find_node_ipython_cloned"] = (
        orig_find_node_ipython_cloned
    )
    orig_find_node_ipython.__globals__["frame_to_node_mapping"] = frame_to_node_mapping
