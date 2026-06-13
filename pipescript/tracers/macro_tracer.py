"""
Subscript-based macro expansion implemented with Pyccolo, to accompany PipelineTracer.
"""

from __future__ import annotations

import ast
import builtins
import re
import textwrap
from contextlib import contextmanager
from functools import reduce
from types import FrameType
from typing import Any, Generator, cast

import pyccolo as pyc
from pyccolo import fast
from pyccolo.stmt_mapper import StatementMapper
from pyccolo.trace_events import TraceEvent
from typing_extensions import Literal

import pipescript.api.static_macros
from pipescript.analysis.dynamic_macros import DynamicMacro
from pipescript.analysis.placeholders import SingletonArgCounterMixin
from pipescript.api.static_macros import (
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


class ArgReplacer(ast.NodeVisitor, SingletonArgCounterMixin):
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
        self.placeholder_nodes.discard(id(node))
        assert node.id.startswith("_")
        if node.id == "_":
            node.id = f"_{self.arg_ctr}"
            self.arg_ctr += 1
        else:
            if not node.id[1].isdigit():
                node.id = node.id[1:]
            else:
                # ensure we prevent collisions with auto placeholders like _1, _2, etc
                node.id = "_" + node.id
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


#: Name of the sentinel builtin a statement-block marker calls (see
#: :func:`block_marker_id` / :func:`_block_marker_sentinel`).
BLOCK_MARKER_FUNC = "__pyc_block__"


def _block_marker_sentinel(_n: int) -> None:
    """Resolve a statement-block marker (``__pyc_block__(N)``) to ``None``.

    The marker is emitted as a call to this *defined* builtin rather than a bare
    (undefined) ``__pyc_block_N__`` name. ipyflow's ``before_subscript_slice``
    machinery evaluates the original slice expression *before* the macro handler
    can substitute it, so an undefined marker name leaks as a ``NameError``
    before ``handle_macro`` ever runs. Calling a defined sentinel evaluates
    harmlessly to ``None``; the handler (which keys off the AST, not the value)
    then fires and replaces the slice with the compiled block, exactly as it
    does for an ordinary ``map[int]``-style slice. (Base pyccolo substitutes
    lazily and never evaluated the marker, which is why this only bit ipyflow.)
    """
    return None


# Register the sentinel as a builtin so the marker call resolves anywhere the
# slice is evaluated (cell globals, compiled-block globals) and survives the
# per-cell tracer ``reset()`` that strips macro-name builtins.
setattr(builtins, BLOCK_MARKER_FUNC, _block_marker_sentinel)


def block_marker_id(node: ast.AST) -> int | None:
    """If ``node`` is a statement-block macro slice (``macro[__pyc_block__(N)]``),
    return ``N``; otherwise ``None``."""
    if not isinstance(node, ast.Subscript):
        return None
    slc = node.slice
    if (
        isinstance(slc, ast.Call)
        and isinstance(slc.func, ast.Name)
        and slc.func.id == BLOCK_MARKER_FUNC
        and len(slc.args) == 1
        and isinstance(slc.args[0], ast.Constant)
        and isinstance(slc.args[0].value, int)
    ):
        return slc.args[0].value
    return None


class _BlockPlaceholderReplacer(ast.NodeVisitor):
    """Identify ``$`` placeholders in a (already ``$``->``_`` transformed)
    statement block by name convention and rename them to concrete parameter
    names, mirroring :class:`ArgReplacer` but for statement bodies whose nodes
    were not position-marked.

    Unlike the expression form -- where each bare ``$`` is a *fresh* positional
    argument -- a statement block treats every bare ``$`` as the single piped
    input (parameter ``_0``), so it can be referenced repeatedly. Use named
    placeholders (``$name``) for additional distinct parameters."""

    def __init__(self) -> None:
        self.auto: list[str] = []
        self.named: list[str] = []

    @property
    def params(self) -> list[str]:
        return self.auto + self.named

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # defer nested macros -- their placeholders belong to them
        if isinstance(node.value, ast.Name) and node.value.id in (
            MacroTracer.static_macros | MacroTracer.dynamic_macros
        ):
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        name = node.id
        if name == "_":
            node.id = "_0"
            if "_0" not in self.auto:
                self.auto.append("_0")
        elif len(name) > 1 and name[0] == "_" and name[1].isalpha():
            pname = name[1:]
            node.id = pname
            if pname not in self.named:
                self.named.append(pname)


def normalize_block_src(src: str) -> str:
    """Normalize a captured block body to consistent indentation.

    The body's first statement is typically *inline* with the opening ``{`` (so
    it carries the column of the ``{``), while later statements carry the source
    indentation -- which an enclosing block's dedent may have shifted relative to
    the first line. ``textwrap.dedent`` can't fix that (the minimum indent may be
    on a later line). Instead we dedent every line by the *first* line's indent,
    clamping at zero, which puts the first statement at column 0 and preserves
    the relative indentation of nested suites beneath it."""
    lines = src.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    base = len(lines[0]) - len(lines[0].lstrip())
    out = []
    for ln in lines:
        if not ln.strip():
            out.append("")
            continue
        indent = len(ln) - len(ln.lstrip())
        out.append(" " * max(0, indent - base) + ln.lstrip())
    return "\n".join(out)


# Forward-pipe operators (``|>`` and friends, with optional ``*`` / ``**``
# unpack prefixes). Their right-hand *stage* expression has its own ``$``
# placeholders (the piped value), which belong to PipelineTracer -- not to the
# block's collapse-``$`` semantics. Matched longest-first so e.g. ``**|>`` wins
# over ``|>``.
_FORWARD_PIPE_RE = re.compile(r"(?:\*\*|\*)?(?:\|>>|\|>|\$>|\?>|\.>)")

# Tokens that end a pipe stage, so a ``$`` after them is the block's again. We
# scope conservatively: commas, comparisons (lower precedence than ``|``), the
# ternary/boolean keywords, and assignment/colon. (The ``<``/``>`` that form
# pipe operators are consumed as pipe spans below, so a bare ``<``/``>`` here is
# a real comparison.)
_STAGE_TERMINATOR_OPS = frozenset(
    {",", "==", "!=", "<", ">", "<=", ">=", "=", ":", ";"}
)
_STAGE_TERMINATOR_KW = frozenset({"and", "or", "not", "in", "is", "if", "else", "for"})


def split_block_placeholders(src: str) -> tuple[str, list[str]]:
    """Replace the *block's own* ``$`` placeholders with parameter names while
    leaving placeholders that belong to nested macros (``macro[...]`` /
    ``macro{...}``) or to nested *pipelines* (the stage after ``|>``) untouched,
    so they are processed by those handlers later.

    Returns ``(new_src, params)``. Bare ``$`` collapses to the single piped
    input ``_0``; ``$name`` becomes a distinct parameter ``name``. A ``$`` is
    "nested" when it sits inside a ``[``/``{`` that immediately follows a macro
    name, or in a forward-pipe stage at the current bracket depth (e.g. the
    second ``$`` in ``$ |> $ + 1`` is the pipe's argument, not the block's)."""
    import io
    import keyword as _kw
    import tokenize as _tok

    macro_names = (
        set(MacroTracer.static_macros)
        | set(MacroTracer.dynamic_macros)
        | set(MacroTracer.dynamic_method_macros)
    )
    try:
        toks = list(_tok.generate_tokens(io.StringIO(src).readline))
    except (_tok.TokenError, IndentationError, SyntaxError):
        return src, []

    line_starts = [0]
    for i, ch in enumerate(src):
        if ch == "\n":
            line_starts.append(i + 1)

    def _off(pos: tuple[int, int]) -> int:
        row = pos[0] - 1
        if row >= len(line_starts):
            return len(src)
        return line_starts[row] + pos[1]

    pipe_spans = [m.span() for m in _FORWARD_PIPE_RE.finditer(src)]

    def _in_pipe_span(start: int, end: int) -> bool:
        return any(s < end and start < e for s, e in pipe_spans)

    repls: list[tuple[int, int, str]] = []
    auto: list[str] = []
    named: list[str] = []
    macro_depth = 0
    opener_is_macro: list[bool] = []
    # one "pipe stage seen" flag per open-bracket depth (index 0 == top level),
    # so a pipe inside ``(...)`` doesn't leak out to siblings.
    pipe_seen: list[bool] = [False]
    prev = None
    for i, t in enumerate(toks):
        if t.type in (
            _tok.NL,
            _tok.NEWLINE,
            _tok.INDENT,
            _tok.DEDENT,
            _tok.COMMENT,
            _tok.ENCODING,
            _tok.ENDMARKER,
        ):
            if t.type in (_tok.NL, _tok.NEWLINE):
                pipe_seen[-1] = False
            continue
        start, end = _off(t.start), _off(t.end)
        if _in_pipe_span(start, end):
            # part of a forward-pipe operator: opens a stage at this depth
            pipe_seen[-1] = True
            prev = t
            continue
        if t.type == _tok.OP and t.string in ("(", "[", "{"):
            if t.string in ("[", "{"):
                is_macro = (
                    prev is not None
                    and prev.type == _tok.NAME
                    and prev.string in macro_names
                    and prev.end == t.start
                )
                opener_is_macro.append(is_macro)
                macro_depth += int(is_macro)
            pipe_seen.append(False)
        elif t.type == _tok.OP and t.string in (")", "]", "}"):
            if t.string in ("]", "}") and opener_is_macro:
                macro_depth -= int(opener_is_macro.pop())
            if len(pipe_seen) > 1:
                pipe_seen.pop()
        elif (t.type == _tok.OP and t.string in _STAGE_TERMINATOR_OPS) or (
            t.type == _tok.NAME and t.string in _STAGE_TERMINATOR_KW
        ):
            pipe_seen[-1] = False
        elif t.string == "$" and macro_depth == 0 and not pipe_seen[-1]:
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if (
                nxt is not None
                and nxt.type == _tok.NAME
                and not _kw.iskeyword(nxt.string)
                and t.end == nxt.start
            ):
                pname = nxt.string
                repls.append((start, _off(nxt.end), pname))
                if pname not in named:
                    named.append(pname)
            else:
                repls.append((start, end, "_0"))
                if "_0" not in auto:
                    auto.append("_0")
        prev = t

    for rstart, rend, text in sorted(repls, reverse=True):
        src = src[:rstart] + text + src[rend:]
    return src, auto + named


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


def is_dynamic_method_macro_attribute(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr in MacroTracer.dynamic_method_macros
    )


def is_dynamic_method_macro(node: ast.AST) -> bool:
    return isinstance(node, ast.Subscript) and is_dynamic_method_macro_attribute(
        node.value
    )


class MacroTracer(pyc.BaseTracer):
    allow_reentrant_events = True
    global_guards_enabled = False
    multiple_threads_allowed = True

    # macros ending in 'f' allow us to write stuff like reduce[+]([1, 2, 3])
    static_macros = {
        context.__name__: context,
        do.__name__: do,
        expect.__name__: expect,
        "f": None,
        fork.__name__: fork,
        future.__name__: future,
        filter.__name__: filter,
        f"i{filter.__name__}": filter,
        f"{filter.__name__}f": filter,
        f"i{filter.__name__}f": filter,
        "macro": None,
        map.__name__: map,
        f"i{map.__name__}": map,
        f"{map.__name__}f": map,
        f"i{map.__name__}f": map,
        memoize.__name__: memoize,
        "method": None,
        ntimes.__name__: ntimes,
        once.__name__: once,
        otherwise.__name__: otherwise,
        parallel.__name__: parallel,
        read.__name__: read,
        reduce.__name__: reduce,
        f"{reduce.__name__}f": reduce,
        repeat.__name__: repeat,
        unless.__name__: unless,
        until.__name__: until,
        when.__name__: when,
        write.__name__: write,
    }

    dynamic_macros: dict[str, DynamicMacro] = {}

    dynamic_method_macros: dict[str, DynamicMacro] = {}

    builtin_dynamic_macro_definitions: dict[str, str] = {
        "foreach": "method[$$ |> map[do[$$]] |> list]",
    }

    assert set(pipescript.api.static_macros.__all__) <= set(static_macros.keys())

    _not_found = object()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arg_replacer = ArgReplacer()
        self.lambda_cache: dict[tuple[int, int, TraceEvent], Any] = {}
        self._overridden_builtins: list[str] = []
        user_ns = get_user_ns()
        for macro_name, macro in (
            self.static_macros | self.dynamic_macros | self.dynamic_method_macros
        ).items():
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

    class _IdentityAttributeSubscript:
        def __getitem__(self, item):
            return item

        def __getattr__(self, item):
            return self

    _identity_attribute_subscript = _IdentityAttributeSubscript()

    @pyc.before_attribute_load(when=is_dynamic_method_macro_attribute, reentrant=True)
    def ignore_dynamic_macro_attribute_load(self, *_, **__):
        return self._identity_attribute_subscript

    @pyc.before_subscript_load(
        when=lambda node: is_dynamic_macro(node) or is_dynamic_method_macro(node),
        reentrant=True,
    )
    def ignore_dynamic_macro_slice(self, *_, **__):
        return self._identity_attribute_subscript

    @pyc.before_subscript_slice(
        when=lambda node: is_dynamic_macro(node) or is_dynamic_method_macro(node),
        reentrant=True,
    )
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
        if isinstance(node.value, ast.Name):
            macro_instance = self.dynamic_macros[node.value.id]
            macro_body = node.slice
        elif isinstance(node.value, ast.Attribute):
            macro_instance = self.dynamic_method_macros[node.value.attr]
            with fast.location_of(node.slice):
                macro_body = fast.Tuple(elts=[node.value.value, node.slice])
        else:
            raise ValueError(
                "Impossible node type for dynamic macro substitution: %s"
                % type(node.value)
            )
        expanded_macro_expr = macro_instance.expand(macro_body)
        if isinstance(expanded_macro_expr, ast.expr):
            lambda_body = expanded_macro_expr
            orig_ctr = self.arg_replacer.arg_ctr
            self.arg_replacer.arg_ctr = orig_ctr
            expanded_macro_expr, *_extra = self.arg_replacer.create_placeholder_lambda(
                [],
                self.arg_replacer.arg_ctr,
                expanded_macro_expr,
                frame,
            )
            expanded_macro_expr.body = lambda_body
            callable_func = pyc.eval(
                expanded_macro_expr, frame.f_globals, frame.f_locals
            )
            try:
                evaluated_lambda = callable_func()
            except Exception as e:
                exc = e

                def raises_error():
                    raise exc

                return lambda: __hide_pyccolo_frame__ and raises_error()
            ret = lambda: __hide_pyccolo_frame__ and evaluated_lambda  # noqa: E731
        else:
            evaluated_lambda = expanded_macro_expr
            ret = lambda: __hide_pyccolo_frame__ and evaluated_lambda  # noqa: E731
        self.lambda_cache[lambda_cache_key] = ret
        return ret

    @pyc.before_subscript_load(when=is_static_macro, reentrant=True)
    def load_macro_result(self, *_, **__):
        return self._identity_attribute_subscript

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
        self,
        orig_lambda_body: ast.expr,
        frame: FrameType,
        func: str,
        allow_call_node: bool = True,
    ) -> ast.expr:
        __hide_pyccolo_frame__ = True  # noqa: F841
        orig_ctr = self.arg_replacer.arg_ctr
        lambda_body = StatementMapper.bookkeeping_propagating_copy(orig_lambda_body)
        placeholder_names = self.arg_replacer.get_placeholder_names(lambda_body)
        needs_call_node = (
            self.arg_replacer.arg_ctr == orig_ctr and len(placeholder_names) == 0
        )
        (
            ast_lambda,
            extra_defaults,
            modified_lambda_body,
        ) = SingletonArgCounterMixin.create_placeholder_lambda(
            placeholder_names,
            orig_ctr,
            lambda_body,
            frame,
            created_starred_arg=needs_call_node,
        )
        lambda_body = modified_lambda_body or lambda_body
        if needs_call_node and allow_call_node:
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

    def _block_free_vars(
        self, body: list[ast.stmt], params: set[str], frame: FrameType
    ) -> set[str]:
        """Names the block reads but never assigns (and that aren't params,
        builtins, or globals); resolved against the defining frame's f_back
        chain. Computed from the placeholder-substituted source before
        instrumentation so pyccolo's own instrumentation names don't leak in."""
        loads: set[str] = set()
        stores: set[str] = set()

        class _NameCollector(ast.NodeVisitor):
            def visit_Name(self, n: ast.Name) -> None:
                (stores if isinstance(n.ctx, ast.Store) else loads).add(n.id)

        collector = _NameCollector()
        for stmt in body:
            collector.visit(stmt)
        return {
            n
            for n in loads - stores - params
            if not hasattr(builtins, n) and not n.startswith("__pyc_")
            # names already visible as globals/builtins resolve directly;
            # `_dynamic_lookup` only walks enclosing *locals*.
            and n not in frame.f_globals
        }

    def _compile_block_function(self, block_src: str, frame: FrameType):
        """Compile a stashed statement block into a function: the block's own
        ``$`` placeholders become parameters, the trailing expression becomes the
        return value, free variables resolve against the defining frame, and
        *nested* pipescript syntax (``|>``, ``f[...]``, ``f{...}``) is parsed,
        marked, and instrumented so it dispatches at runtime.

        The body is compiled via :meth:`~pyccolo.BaseTracer.parse_fragment`,
        which instruments the fragment from inside this active macro expansion
        without re-entering / corrupting it."""
        # Hidden so a co-tracer (e.g. ipyflow) that walks the frame stack to map
        # a sandbox call back to a notebook position skips this frame instead of
        # misattributing its own line number to the executing cell.
        __hide_pyccolo_frame__ = True  # noqa: F841
        from pipescript.analysis.placeholders import FreeVarTransformer
        from pipescript.api.utils import _dynamic_lookup

        block_src = normalize_block_src(block_src)
        # Substitute the block's own placeholders for parameters, leaving nested
        # macros' placeholders (still `$`) for pyccolo to mark/instrument.
        param_src, params = split_block_placeholders(block_src)
        param_src = normalize_block_src(param_src)
        # Free-var analysis on a clean (uninstrumented) parse of the same source.
        clean_body = ast.parse(textwrap.dedent(self.transform(param_src)).strip("\n"))
        freevars = self._block_free_vars(clean_body.body, set(params), frame)

        # Parse the body as a *module* (at column 0) and wrap it in a FunctionDef
        # at the AST level. Wrapping in `def ...:` *textually* (with
        # textwrap.indent) corrupts a nested block whose first statement is inline
        # with `{` -- its lines end up inconsistently indented and fail to parse.
        #
        # Scope the instrumentation to the substituting tracers (those with
        # ``global_guards_enabled = False`` -- i.e. pipescript's own, which the
        # block's nested macros need), excluding pure observers like ipyflow's
        # dataflow tracer. The block is synthetic sandbox code an observer cannot
        # map back to a notebook statement; weaving its statement events in makes
        # the observer raise on the block's unknown nodes.
        from pyccolo.tracer import _TRACER_STACK

        block_tracers = [
            t for t in _TRACER_STACK if not t.global_guards_enabled
        ] or None
        body_module = cast(
            ast.Module, self.parse_fragment(param_src, tracers=block_tracers)
        )
        body = list(body_module.body)
        # instrumentation may wrap the body in a try/except; the trailing
        # expression we want to return lives at the innermost level.
        target_body = body
        while len(target_body) == 1 and isinstance(target_body[0], ast.Try):
            target_body = target_body[0].body
        if target_body and isinstance(target_body[-1], ast.Expr):
            target_body[-1] = ast.Return(value=target_body[-1].value)
        if freevars:
            transformer = FreeVarTransformer(freevars, frame)
            body = [transformer.visit(stmt) for stmt in body]

        fn_def = cast(
            ast.FunctionDef,
            ast.parse("def __pyc_macro_block__(*__pyc_rest__):\n    pass").body[0],
        )
        fn_def.args.args = [ast.arg(arg=p) for p in params]
        fn_def.body = body or [ast.Pass()]
        module = ast.Module(body=[fn_def], type_ignores=[])
        ast.fix_missing_locations(module)

        block_globals = dict(frame.f_globals)
        block_globals.setdefault("_dynamic_lookup", _dynamic_lookup)
        local_ns: dict = {}
        # exec_raw with instrument=False reuses the active trace (no reset) so
        # the already-instrumented nested macros still fire.
        self.exec_raw(
            module,
            global_env=block_globals,
            local_env=local_ns,
            filename=self.make_sandbox_fname(),
            instrument=False,
        )
        return local_ns["__pyc_macro_block__"]

    def _handle_block_macro(
        self, node: ast.Subscript, frame: FrameType, func: str, block_id: int
    ) -> ast.expr:
        __hide_pyccolo_frame__ = True  # noqa: F841
        from pipescript.tracers.brace_block_tracer import BraceBlockTracer

        block_fn = self._compile_block_function(
            BraceBlockTracer.block_sources[block_id], frame
        )
        # expose the compiled function so the macro wrapper can reference it
        gname = f"__pyc_block_fn_{id(block_fn)}__"
        frame.f_globals[gname] = block_fn
        with fast.location_of(node):
            ast_lambda: ast.expr = fast.Name(gname, ctx=ast.Load())
            if func in self.static_macros and func not in (
                "f",
                memoize.__name__,
                otherwise.__name__,
            ):
                ast_lambda = self._transform_ast_lambda_for_macro(
                    ast_lambda, func, set()
                )
        return ast_lambda

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
        block_id = block_marker_id(node)
        if block_id is not None:
            callable_expr = self._handle_block_macro(node, frame, func, block_id)
            evaluated_lambda = pyc.eval(callable_expr, frame.f_globals, frame.f_locals)
            ret = lambda: __hide_pyccolo_frame__ and evaluated_lambda  # noqa: E731
            self.lambda_cache[lambda_cache_key] = ret
            return ret
        callable_expr: ast.expr
        if func in ("macro", "method"):
            macro = DynamicMacro.create(node.slice, self, is_method=func == "method")
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
