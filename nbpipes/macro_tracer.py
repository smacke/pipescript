"""
Subscript-based macro expansion implemented with Pyccolo, to accompany PipelineTracer.
"""

from __future__ import annotations

import ast
import builtins
from functools import reduce
from types import FrameType
from typing import Any, cast

import pyccolo as pyc
from pyccolo import fast
from pyccolo.stmt_mapper import StatementMapper
from pyccolo.trace_events import TraceEvent

from nbpipes.pipeline_tracer import PipelineTracer, SingletonArgCounterMixin


class _ArgReplacer(ast.NodeVisitor, SingletonArgCounterMixin):
    def __init__(self) -> None:
        super().__init__()
        self.placeholder_names: dict[str, None] = {}

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in MacroTracer.macros:
            # defer visiting nested quick lambdas
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if (
            node.id != "_"
            and id(node)
            not in PipelineTracer.augmented_node_ids_by_spec[
                PipelineTracer.arg_placeholder_spec
            ]
        ):
            return
        # quick lambda will interpret this node as placeholder without any aug spec necessary
        PipelineTracer.augmented_node_ids_by_spec[
            PipelineTracer.arg_placeholder_spec
        ].discard(id(node))
        assert node.id.startswith("_")
        if node.id == "_":
            node.id = f"_{self.arg_ctr}"
            self.arg_ctr += 1
        else:
            if node.id[1].isalpha():
                node.id = node.id[1:]
            self.placeholder_names[node.id] = None

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.BitOr) and PipelineTracer.get_augmentations(
            id(node)
        ):
            return
        self.generic_visit(node)

    def get_placeholder_names(self, node: ast.AST) -> list[str]:
        self.placeholder_names.clear()
        self.visit(node)
        return list(self.placeholder_names)


class _IdentitySubscript:
    def __getitem__(self, item):
        return item


_identity_subscript = _IdentitySubscript()


def is_macro(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in MacroTracer.macros
    )


class MacroTracer(pyc.BaseTracer):

    allow_reentrant_events = True
    global_guards_enabled = False

    macros = ("f", "filter", "ifilter", "map", "imap", "reduce")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._extra_builtins: set[str] = set()
        self._arg_replacer = _ArgReplacer()
        builtins.reduce = reduce  # type: ignore[attr-defined]
        builtins.imap = map  # type: ignore[attr-defined]
        self.lambda_cache: dict[tuple[int, int, TraceEvent], Any] = {}

    def enter_tracing_hook(self) -> None:
        import builtins

        # need to create dummy reference to avoid NameError
        for macro in self.macros:
            if not hasattr(builtins, macro):
                self._extra_builtins.add(macro)
                setattr(builtins, macro, None)

    def exit_tracing_hook(self) -> None:
        import builtins

        for macro in self._extra_builtins:
            if hasattr(builtins, macro):
                delattr(builtins, macro)
        self._extra_builtins.clear()

    @pyc.before_subscript_load(when=is_macro, reentrant=True)
    def load_macro_result(self, _ret, *_, **__):
        return _identity_subscript

    _not_found = object()

    @pyc.before_subscript_slice(when=is_macro, reentrant=True)
    def handle_quick_lambda(
        self, _ret, node: ast.Subscript, frame: FrameType, evt: TraceEvent, *_, **__
    ):
        lambda_cache_key = (id(node), id(frame), evt)
        cached_lambda = self.lambda_cache.get(lambda_cache_key, self._not_found)
        if cached_lambda is not self._not_found:
            return cached_lambda
        __hide_pyccolo_frame__ = True
        orig_ctr = self._arg_replacer.arg_ctr
        orig_lambda_body: ast.expr = node.slice  # type: ignore[assignment]
        if isinstance(orig_lambda_body, ast.Index):
            orig_lambda_body = orig_lambda_body.value  # type: ignore[attr-defined]
        lambda_body = StatementMapper.augmentation_propagating_copy(orig_lambda_body)
        placeholder_names = self._arg_replacer.get_placeholder_names(lambda_body)
        if self._arg_replacer.arg_ctr == orig_ctr and len(placeholder_names) == 0:
            ast_lambda = lambda_body
        else:
            ast_lambda = SingletonArgCounterMixin.create_placeholder_lambda(
                placeholder_names, orig_ctr, lambda_body, frame.f_globals
            )
            ast_lambda.body = lambda_body
        func = cast(ast.Name, node.value).id
        if func in ("filter", "ifilter", "map", "imap", "reduce"):
            with fast.location_of(ast_lambda):
                arg = f"_{self._arg_replacer.arg_ctr}"
                self._arg_replacer.arg_ctr += 1
                inner_func = func
                if func == "ifilter":
                    inner_func = "filter"
                elif func == "imap":
                    inner_func = "map"
                lambda_body_str = f"{inner_func}(None, {arg})"
                functor_lambda_body = cast(
                    ast.Call,
                    cast(
                        ast.Expr,
                        fast.parse(lambda_body_str).body[0],
                    ).value,
                )
                functor_lambda_body.args[0] = ast_lambda
                if func in ("filter", "map"):
                    id_arg = f"_{self._arg_replacer.arg_ctr}"
                    self._arg_replacer.arg_ctr += 1
                    lambda_body_str = f"(list if type({arg}) is list else lambda {id_arg}: {id_arg})(None)"
                    functor_lambda_outer_body = cast(
                        ast.Call,
                        cast(
                            ast.Expr,
                            fast.parse(lambda_body_str).body[0],
                        ).value,
                    )
                    functor_lambda_outer_body.args[0] = functor_lambda_body
                    functor_lambda_body = functor_lambda_outer_body
                functor_lambda = cast(
                    ast.Lambda,
                    cast(ast.Expr, fast.parse(f"lambda {arg}: None").body[0]).value,
                )
                functor_lambda.body = functor_lambda_body
            ast_lambda = functor_lambda
        evaluated_lambda = pyc.eval(ast_lambda, frame.f_globals, frame.f_locals)
        ret = lambda: __hide_pyccolo_frame__ and evaluated_lambda  # noqa: E731
        self.lambda_cache[lambda_cache_key] = ret
        return ret
