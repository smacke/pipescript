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

from nbpipes.pipeline_tracer import PipelineTracer
from nbpipes.placeholders import SingletonArgCounterMixin
from nbpipes.utils import do, fork, get_user_ns


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

    def visit_pipeline(self, node: ast.BinOp) -> None:
        num_left_traversals = PipelineTracer.search_left_descendant_placeholder(node)
        if num_left_traversals < 0:
            return
        left_arg: ast.expr = node
        for _ in range(num_left_traversals):
            left_arg = node.left
        self.visit(left_arg)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.BitOr) and PipelineTracer.get_augmentations(
            id(node)
        ):
            self.visit_pipeline(node)
        else:
            self.generic_visit(node)

    def get_placeholder_names(self, node: ast.AST) -> list[str]:
        self.placeholder_names.clear()
        self.visit(node)
        return list(self.placeholder_names)


def is_macro(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in MacroTracer.macros
    )


class MacroTracer(pyc.BaseTracer):

    allow_reentrant_events = True
    global_guards_enabled = False

    macros = {
        do.__name__: do,
        "f": None,
        fork.__name__: fork,
        filter.__name__: filter,
        "ifilter": filter,
        map.__name__: map,
        "imap": map,
        reduce.__name__: reduce,
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arg_replacer = _ArgReplacer()
        self.lambda_cache: dict[tuple[int, int, TraceEvent], Any] = {}
        with self.register_additional_ast_bookkeeping():
            self.placeholder_inference_skip_nodes: set[int] = set()
        user_ns = get_user_ns()
        for macro_name, macro in self.macros.items():
            if hasattr(builtins, macro_name):
                continue
            setattr(builtins, macro_name, macro)
            if user_ns is not None:
                user_ns[macro_name] = macro

    class _IdentitySubscript:
        def __getitem__(self, item):
            return item

    _identity_subscript = _IdentitySubscript()

    @pyc.before_subscript_load(when=is_macro, reentrant=True)
    def load_macro_result(self, _ret, *_, **__):
        return self._identity_subscript

    _not_found = object()

    def _transform_ast_lambda_for_macro(
        self, ast_lambda: ast.Lambda, func: str
    ) -> ast.Lambda:
        arg = f"_{self.arg_replacer.arg_ctr}"
        self.arg_replacer.arg_ctr += 1
        starred_arg = f"_{self.arg_replacer.arg_ctr}"
        self.arg_replacer.arg_ctr += 1
        inner_func = func
        if func == "ifilter":
            inner_func = "filter"
        elif func == "imap":
            inner_func = "map"
        lambda_body_str = f"{inner_func}(None, {arg}, *{starred_arg})"
        functor_lambda_body = cast(
            ast.Call,
            cast(
                ast.Expr,
                fast.parse(lambda_body_str).body[0],
            ).value,
        )
        functor_lambda_body.args[0] = ast_lambda
        if func in ("filter", "map"):
            id_arg = f"_{self.arg_replacer.arg_ctr}"
            self.arg_replacer.arg_ctr += 1
            lambda_body_str = f"(type({arg}) if type({arg}) in (frozenset, list, set, tuple) else lambda {id_arg}: {id_arg})(None)"
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
            cast(
                ast.Expr,
                fast.parse(f"lambda {arg}, *{starred_arg}: None").body[0],
            ).value,
        )
        functor_lambda.body = functor_lambda_body
        return functor_lambda

    def _handle_macro_impl(
        self, orig_lambda_body: ast.expr, frame: FrameType, func: str
    ) -> ast.Lambda:
        __hide_pyccolo_frame__ = True  # noqa: F841
        orig_ctr = self.arg_replacer.arg_ctr
        lambda_body = StatementMapper.bookkeeping_propagating_copy(orig_lambda_body)
        placeholder_names = self.arg_replacer.get_placeholder_names(lambda_body)
        if self.arg_replacer.arg_ctr == orig_ctr and len(placeholder_names) == 0:
            ast_lambda = lambda_body
        else:
            ast_lambda = SingletonArgCounterMixin.create_placeholder_lambda(
                placeholder_names, orig_ctr, lambda_body, frame.f_globals
            )
            ast_lambda.body = lambda_body
        if func in self.macros and func != "f":
            with fast.location_of(ast_lambda):
                ast_lambda = self._transform_ast_lambda_for_macro(ast_lambda, func)
        return ast_lambda

    @pyc.before_subscript_slice(when=is_macro, reentrant=True)
    def handle_macro(
        self, _ret, node: ast.Subscript, frame: FrameType, evt: TraceEvent, *_, **__
    ):
        lambda_cache_key = (id(node), id(frame), evt)
        cached_lambda = self.lambda_cache.get(lambda_cache_key, self._not_found)
        if cached_lambda is not self._not_found:
            return cached_lambda
        __hide_pyccolo_frame__ = True
        func = cast(ast.Name, node.value).id
        if func == fork.__name__ and isinstance(node.slice, ast.Tuple):
            lambdas: list[ast.Lambda] = []
            max_nargs = 1
            for expr in node.slice.elts:
                expr_lambda = self._handle_macro_impl(expr, frame, "f")
                lambdas.append(expr_lambda)
                if isinstance(expr_lambda, ast.Lambda):
                    max_nargs = max(max_nargs, len(expr_lambda.args.args))
            with fast.location_of(node.slice):
                args = [
                    f"_{arg_ctr}"
                    for arg_ctr in range(
                        self.arg_replacer.arg_ctr, self.arg_replacer.arg_ctr + max_nargs
                    )
                ]
                arg_str = ", ".join(args)
                self.arg_replacer.arg_ctr += max_nargs
                ast_lambda = fast.parse(f"lambda {arg_str}: None").body[0].value
                load = ast.Load()
                tuple_elts: list[ast.Call] = []
                for lam in lambdas:
                    call_node = fast.Call(
                        func=lam,
                        args=[fast.Name(arg, ctx=load) for arg in args],
                        keywords=[],
                    )
                    self.placeholder_inference_skip_nodes.add(id(call_node))
                    tuple_elts.append(call_node)
                ast_lambda.body = fast.Tuple(tuple_elts, ctx=load)
        else:
            ast_lambda = self._handle_macro_impl(node.slice, frame, func)
        evaluated_lambda = pyc.eval(ast_lambda, frame.f_globals, frame.f_locals)
        ret = lambda: __hide_pyccolo_frame__ and evaluated_lambda  # noqa: E731
        self.lambda_cache[lambda_cache_key] = ret
        return ret
