"""
Subscript-based macro expansion implemented with Pyccolo, to accompany PipelineTracer.
"""

from __future__ import annotations

import ast
import builtins
from contextlib import contextmanager
from functools import reduce
from types import FrameType
from typing import Any, Generator, cast

import pyccolo as pyc
from pyccolo import fast
from pyccolo.stmt_mapper import StatementMapper
from pyccolo.trace_events import TraceEvent
from typing_extensions import Literal

import pipescript.api.macros
from pipescript.analysis.placeholders import SingletonArgCounterMixin
from pipescript.api.macros import (
    _ntimes_counters,
    _once_cache,
    context,
    do,
    expect,
    fork,
    future,
    memoize,
    ntimes,
    once,
    otherwise,
    parallel,
    read,
    repeat,
    unless,
    until,
    when,
    write,
)
from pipescript.tracers.pipeline_tracer import PipelineTracer
from pipescript.utils import get_user_ns


class _ArgReplacer(ast.NodeVisitor, SingletonArgCounterMixin):
    def __init__(self) -> None:
        super().__init__()
        self.placeholder_names: list[str] = []
        self.arg_node_id_to_placeholder_name: dict[int, str] = {}
        self._macro_visit_context: bool = False

    @contextmanager
    def macro_visit_context(self, override: bool = True) -> Generator[None, None, None]:
        old = self._macro_visit_context
        try:
            self._macro_visit_context = override
            yield
        finally:
            self._macro_visit_context = old

    @property
    def placeholder_nodes(self) -> set[int]:
        if self._macro_visit_context:
            return PipelineTracer.augmented_node_ids_by_spec[
                PipelineTracer.macro_arg_placeholder_spec
            ]
        else:
            return PipelineTracer.augmented_node_ids_by_spec[
                PipelineTracer.arg_placeholder_spec
            ]

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            not self._macro_visit_context
            and isinstance(node.value, ast.Name)
            and node.value.id in MacroTracer.static_macros
        ):
            # defer visiting nested quick lambdas
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if id(node) not in self.placeholder_nodes:
            return
        self.placeholder_nodes.remove(id(node))
        assert node.id.startswith("_")
        if node.id == "_":
            node.id = f"_{self.arg_ctr}"
            self.arg_ctr += 1
        else:
            if node.id[1].isalpha():
                node.id = node.id[1:]
            if node.id not in self.placeholder_names:
                self.placeholder_names.append(node.id)
        self.arg_node_id_to_placeholder_name[id(node)] = node.id

    def visit_pipeline(self, node: ast.BinOp) -> None:
        num_left_traversals = PipelineTracer.search_left_descendant_placeholder(node)
        if num_left_traversals < 0:
            return
        left_arg: ast.expr = node
        for _ in range(num_left_traversals):
            left_arg = left_arg.left  # type: ignore[attr-defined]
        self.visit(left_arg)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if (
            not self._macro_visit_context
            and isinstance(node.op, ast.BitOr)
            and PipelineTracer.get_augmentations(id(node))
        ):
            self.visit_pipeline(node)
        else:
            self.generic_visit(node)

    def __call__(self, node: ast.AST) -> None:
        self.arg_node_id_to_placeholder_name.clear()
        self.placeholder_names.clear()
        self.visit(node)

    def get_placeholder_names(self, node: ast.AST) -> list[str]:
        self(node)
        return self.placeholder_names


def is_static_macro(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in MacroTracer.static_macros
    )


def is_dynamic_macro(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in MacroTracer.dynamic_macros
    )


class DynamicMacroArgSubstitutor(ast.NodeTransformer):
    def __init__(
        self,
        arg_node_id_to_placeholder_name: dict[int, str],
        ordered_arg_names: list[str],
        arg_node_subst_exprs: list[ast.expr],
    ) -> None:
        self.arg_node_id_to_placeholder_name = arg_node_id_to_placeholder_name
        self.ordered_arg_names = ordered_arg_names
        self.arg_node_subst_exprs = arg_node_subst_exprs

    def visit_Name(self, node: ast.Name) -> ast.AST:
        arg_name = self.arg_node_id_to_placeholder_name.get(id(node))
        if arg_name is None:
            return super().generic_visit(node)
        return self.arg_node_subst_exprs[self.ordered_arg_names.index(arg_name)]


class DynamicMacro:
    def __init__(self, template: ast.expr, ordered_arg_names: list[str]) -> None:
        self.template = template
        self.ordered_arg_names = ordered_arg_names

    def expand(self, args: ast.expr) -> ast.expr:
        template_copy: ast.expr = StatementMapper.bookkeeping_propagating_copy(
            self.template
        )
        arg_replacer = MacroTracer.instance().arg_replacer
        with arg_replacer.macro_visit_context():
            arg_replacer(template_copy)
        arg_node_id_to_placeholder_name = arg_replacer.arg_node_id_to_placeholder_name
        ordered_arg_names = list(self.ordered_arg_names)
        for arg_name in arg_node_id_to_placeholder_name.values():
            if arg_name not in ordered_arg_names:
                ordered_arg_names.append(arg_name)
        expanded_args: list[ast.expr] = [args]
        if len(ordered_arg_names) > 1:
            if not isinstance(args, (ast.List, ast.Tuple)):
                raise ValueError(
                    "Need multiple args but unable to expand singleton subscript arg"
                )
            expanded_args = args.elts
        if len(ordered_arg_names) != len(expanded_args):
            raise ValueError(
                "Wrong number of arguments to macro: expected %d but got %d"
                % (len(ordered_arg_names), len(expanded_args))
            )
        substitutor = DynamicMacroArgSubstitutor(
            arg_node_id_to_placeholder_name, ordered_arg_names, expanded_args
        )
        return substitutor.visit(template_copy)


class MacroTracer(pyc.BaseTracer):

    allow_reentrant_events = True
    global_guards_enabled = False
    multiple_threads_allowed = True

    static_macros = {
        context.__name__: context,
        do.__name__: do,
        expect.__name__: expect,
        "f": None,
        fork.__name__: fork,
        future.__name__: future,
        filter.__name__: filter,
        "ifilter": filter,
        "macro": None,
        map.__name__: map,
        "imap": map,
        memoize.__name__: memoize,
        ntimes.__name__: ntimes,
        once.__name__: once,
        otherwise.__name__: otherwise,
        parallel.__name__: parallel,
        read.__name__: read,
        reduce.__name__: reduce,
        repeat.__name__: repeat,
        unless.__name__: unless,
        until.__name__: until,
        when.__name__: when,
        write.__name__: write,
    }

    dynamic_macros: dict[str, DynamicMacro] = {}

    assert set(pipescript.api.macros.__all__) <= set(static_macros.keys())

    _not_found = object()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arg_replacer = _ArgReplacer()
        self.lambda_cache: dict[tuple[int, int, TraceEvent], Any] = {}
        self._overridden_builtins: list[str] = []
        user_ns = get_user_ns()
        for macro_name, macro in (self.static_macros | self.dynamic_macros).items():
            if hasattr(builtins, macro_name):
                continue
            setattr(builtins, macro_name, macro)
            self._overridden_builtins.append(macro_name)
            if user_ns is not None:
                user_ns.setdefault(macro_name, macro)

    def reset(self) -> None:
        for macro_name in self._overridden_builtins:
            if hasattr(builtins, macro_name):
                delattr(builtins, macro_name)
        self._overridden_builtins.clear()
        super().reset()

    class _IdentitySubscript:
        def __getitem__(self, item):
            return item

    _identity_subscript = _IdentitySubscript()

    @pyc.before_subscript_load(when=is_dynamic_macro, reentrant=True)
    def ignore_dynamic_macro_slice(self, *_, **__):
        return self._identity_subscript

    @pyc.before_subscript_slice(when=is_dynamic_macro, reentrant=True)
    def perform_dynamic_macro_substitution(
        self,
        _ret,
        node: ast.Subscript,
        frame: FrameType,
        evt: TraceEvent,
        *_,
        **__,
    ):
        lambda_cache_key = (id(node), id(frame), evt)
        cached_lambda = self.lambda_cache.get(lambda_cache_key, self._not_found)
        if cached_lambda is not self._not_found:
            return cached_lambda
        __hide_pyccolo_frame__ = True
        assert isinstance(node.value, ast.Name)
        macro_instance = self.dynamic_macros[node.value.id]
        expanded_macro_expr = macro_instance.expand(node.slice)
        evaluated_lambda = pyc.eval(
            expanded_macro_expr, frame.f_globals, frame.f_locals
        )
        ret = lambda: __hide_pyccolo_frame__ and evaluated_lambda  # noqa: E731
        self.lambda_cache[lambda_cache_key] = ret
        return ret

    @pyc.before_subscript_load(when=is_static_macro, reentrant=True)
    def load_macro_result(self, *_, **__):
        return self._identity_subscript

    def _transform_ast_lambda_for_macro(
        self,
        ast_lambda: ast.expr,
        func: str,
        extra_defaults: set[str],
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
                fast.parse(
                    f"lambda {arg}, *{starred_arg}, {', '.join(f'{name}={name}' for name in extra_defaults)}: None"
                ).body[0],
            ).value,
        )
        functor_lambda.body = functor_lambda_body
        return functor_lambda

    def _handle_macro_impl(
        self, orig_lambda_body: ast.expr, frame: FrameType, func: str
    ) -> ast.expr:
        __hide_pyccolo_frame__ = True  # noqa: F841
        orig_ctr = self.arg_replacer.arg_ctr
        lambda_body = StatementMapper.bookkeeping_propagating_copy(orig_lambda_body)
        placeholder_names = self.arg_replacer.get_placeholder_names(lambda_body)
        needs_call_node = (
            self.arg_replacer.arg_ctr == orig_ctr and len(placeholder_names) == 0
        )
        ast_lambda, extra_defaults, modified_lambda_body = (
            SingletonArgCounterMixin.create_placeholder_lambda(
                placeholder_names,
                orig_ctr,
                lambda_body,
                frame,
                created_starred_arg=needs_call_node,
            )
        )
        lambda_body = modified_lambda_body or lambda_body
        if needs_call_node:
            with fast.location_of(lambda_body):
                load = ast.Load()
                lambda_body = fast.Call(
                    func=lambda_body,
                    args=[
                        fast.Starred(
                            fast.Name(f"_{self.arg_replacer.arg_ctr - 1}", ctx=load),
                            ctx=load,
                        )
                    ],
                    keywords=[],
                )
        ast_lambda.body = lambda_body
        if func in self.static_macros and func not in (
            "f",
            memoize.__name__,
            otherwise.__name__,
        ):
            with fast.location_of(ast_lambda):
                ast_lambda = self._transform_ast_lambda_for_macro(
                    ast_lambda, func, extra_defaults
                )
        ret_expr: ast.expr = ast_lambda
        if func in (memoize.__name__, otherwise.__name__):
            with fast.location_of(ast_lambda):
                ret_expr = fast.Call(
                    func=fast.Name(func, ctx=ast.Load()),
                    args=[ast_lambda],
                    keywords=[],
                )
        return ret_expr

    @fast.location_of_arg
    def _handle_read_write_macro(
        self, func: Literal["read", "write"], macro_body: ast.expr
    ) -> ast.Lambda:
        if isinstance(macro_body, ast.Name):
            macro_body = fast.Str(macro_body.id)
        arg = f"_{self.arg_replacer.arg_ctr}"
        self.arg_replacer.arg_ctr += 1
        lam: ast.Lambda = cast(
            ast.Lambda,
            cast(
                ast.Expr, fast.parse(f"lambda {arg}: {func}(None, {arg})").body[0]
            ).value,
        )
        cast(ast.Call, lam.body).args[0] = macro_body
        return lam

    @fast.location_of_arg
    def _handle_ntimes_macro(
        self, frame: FrameType, macro_body: ast.expr
    ) -> ast.Lambda:
        callpoint_id = id(macro_body)
        if callpoint_id not in _ntimes_counters:
            if isinstance(macro_body, ast.Constant) and isinstance(
                macro_body.value, int
            ):
                ctr = macro_body.value
            else:
                ctr = pyc.eval(macro_body, frame.f_globals, frame.f_locals)
            _ntimes_counters[callpoint_id] = ctr
        arg = f"_{self.arg_replacer.arg_ctr}"
        self.arg_replacer.arg_ctr += 1
        lam: ast.Lambda = cast(
            ast.Lambda,
            cast(
                ast.Expr,
                fast.parse(f"lambda {arg}: {ntimes.__name__}({arg}, None)").body[0],
            ).value,
        )
        cast(ast.Call, lam.body).args[1] = fast.Constant(value=callpoint_id)
        return lam

    @fast.location_of_arg
    def _handle_once_macro(self, frame: FrameType, macro_body: ast.expr) -> ast.expr:
        callpoint_id = id(macro_body)
        if callpoint_id not in _once_cache:
            _once_cache[callpoint_id] = pyc.eval(
                macro_body, frame.f_globals, frame.f_locals
            )
        expr = cast(
            ast.Expr,
            fast.parse(f"{once.__name__}(None)").body[0],
        ).value
        cast(ast.Call, expr).args[0] = fast.Constant(value=callpoint_id)
        return expr

    @pyc.before_subscript_slice(when=is_static_macro, reentrant=True)
    def handle_macro(
        self, _ret, node: ast.Subscript, frame: FrameType, evt: TraceEvent, *_, **__
    ):
        lambda_cache_key = (id(node), id(frame), evt)
        cached_lambda = self.lambda_cache.get(lambda_cache_key, self._not_found)
        if cached_lambda is not self._not_found:
            return cached_lambda
        __hide_pyccolo_frame__ = True
        func = cast(ast.Name, node.value).id
        callable_expr: ast.expr
        if func == "macro":
            if (
                isinstance(node.slice, ast.Tuple)
                and len(node.slice.elts) > 1
                and all(isinstance(elt, ast.Name) for elt in node.slice.elts[1:])
            ):
                macro_template = node.slice.elts[0]
                ordered_arg_names = [
                    cast(ast.Name, elt).id for elt in node.slice.elts[1:]
                ]
            else:
                macro_template = node.slice
                ordered_arg_names = []
            macro = DynamicMacro(macro_template, ordered_arg_names)
            ret = lambda: __hide_pyccolo_frame__ and macro  # noqa: E731
            self.lambda_cache[lambda_cache_key] = ret
            return ret
        elif func in (fork.__name__, parallel.__name__) and isinstance(
            node.slice, ast.Tuple
        ):
            callables: list[ast.expr] = []
            max_nargs = 1
            expr: ast.expr | None = None
            for expr in node.slice.elts:
                expr_lambda = self._handle_macro_impl(expr, frame, "f")
                callables.append(expr_lambda)
                if isinstance(expr_lambda, ast.Lambda):
                    max_nargs = max(max_nargs, len(expr_lambda.args.args))
            has_otherwise = (
                isinstance(expr, ast.Subscript)
                and isinstance(expr.value, ast.Name)
                and expr.value.id == otherwise.__name__
            )
            with fast.location_of(node.slice):
                args = [
                    f"_{arg_ctr}"
                    for arg_ctr in range(
                        self.arg_replacer.arg_ctr, self.arg_replacer.arg_ctr + max_nargs
                    )
                ]
                arg_str = ", ".join(args)
                self.arg_replacer.arg_ctr += max_nargs
                ast_lambda = cast(
                    ast.Lambda,
                    cast(ast.Expr, fast.parse(f"lambda {arg_str}: None").body[0]).value,
                )
                load = ast.Load()
                tuple_elts: list[ast.expr] = []
                if has_otherwise:
                    tuple_elts.append(fast.Constant(value=True))
                for lam in callables:
                    tuple_elts.append(lam)
                ast_lambda.body = fast.Call(
                    func=fast.Name(id=func, ctx=load),
                    args=[fast.Tuple(tuple_elts, ctx=load)]
                    + [fast.Name(arg, ctx=load) for arg in args],
                    keywords=[],
                )
                callable_expr = ast_lambda
        elif func in (read.__name__, write.__name__):
            rw_lambda = self._handle_read_write_macro(func, node.slice)  # type: ignore[arg-type]
            callable_expr = cast(ast.expr, rw_lambda)
        elif func == ntimes.__name__:
            ntimes_lambda = self._handle_ntimes_macro(frame, node.slice)  # type: ignore[arg-type]
            callable_expr = cast(ast.expr, ntimes_lambda)
        elif func == once.__name__:
            once_call = self._handle_once_macro(frame, node.slice)
            callable_expr = cast(ast.expr, once_call)
        else:
            callable_expr = self._handle_macro_impl(node.slice, frame, func)
        evaluated_lambda = pyc.eval(callable_expr, frame.f_globals, frame.f_locals)
        ret = lambda: __hide_pyccolo_frame__ and evaluated_lambda  # noqa: E731
        self.lambda_cache[lambda_cache_key] = ret
        return ret
