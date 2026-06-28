"""
Brace syntax for pipescript macros, including multi-line *statement* bodies.

Lets any macro that is normally written ``macro[ ... ]`` also be written
``macro{ ... }``. Two cases:

* **Expression body** (``f{ $ + 1 }``): the braces are swapped for brackets
  inline (``f[ $ + 1 ]``), so the ordinary ``$`` placeholder machinery applies
  unchanged.
* **Statement body** (``f{ acc = 0; for i in range($): acc += i; acc }``): the
  statements can't live in a subscript slice, so the raw source is stashed
  out-of-band and the slice is replaced with a marker, ``f[__pyc_block_<N>__]``.
  :class:`~pipescript.tracers.macro_tracer.MacroTracer` compiles the stashed
  statements into a function at expansion time -- ``$`` placeholders become the
  function's parameters, exactly like the expression case, and the trailing
  expression becomes the return value.

The rewrite rides pyccolo's custom-augmentation framework (a
:class:`~pyccolo.CustomRewrite` carried by :data:`BraceBlockTracer.brace_spec`)
rather than a bespoke ``preprocess`` / ``make_syntax_augmenter`` override. That
means it threads source positions through the rewrite (the old override silently
dropped them) and round-trips through ``untransform`` -- ``macro[...]`` resugars
back to ``macro{...}`` -- instead of needing a parallel regex resugarer.

Enter this tracer **outermost** (before ``PipelineTracer``) so the brace
extraction happens before the ``$`` -> ``_`` placeholder pass; that keeps the
captured body's ``$`` intact (re-processed later) and avoids registering stale
placeholder positions for source that has been moved out of band.
"""

from __future__ import annotations

import ast
import threading
from collections.abc import MutableMapping
from typing import Callable, Iterator, TypeVar

import pyccolo as pyc
from pyccolo.syntax_augmentation import (
    Range,
    _find_first_paired_construct,
    _line_starts,
    line_col_of,
    offset_of,
)

_K = TypeVar("_K")
_V = TypeVar("_V")


class _ThreadLocalDict(MutableMapping[_K, _V]):
    """A dict-like mapping with independent per-thread storage.

    ``block_sources`` / ``_id_by_source`` are written and read on the execution
    thread, but a second consumer (e.g. an in-kernel language server linting on a
    background thread) also drives the brace augmenter against the live tracer
    singletons. Per-thread storage gives each thread its own body registry, so an
    analysis pass on one thread can't overwrite the body a concurrent execution
    registered on another. ipyflow's several augmenter passes per cell all run on
    the same (execution) thread, so they still share one store -- the
    ``_id_by_source`` dedup stays idempotent within a cell.

    ``_counter`` deliberately stays a shared class int: it only mints ids, and
    once the maps it indexes are per-thread a duplicated id can't cross threads.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def _store(self) -> "dict[_K, _V]":
        store = getattr(self._local, "store", None)
        if store is None:
            store = self._local.store = {}
        return store

    def __getitem__(self, key: _K) -> _V:
        return self._store()[key]

    def __setitem__(self, key: _K, value: _V) -> None:
        self._store()[key] = value

    def __delitem__(self, key: _K) -> None:
        del self._store()[key]

    def __iter__(self) -> Iterator[_K]:
        return iter(self._store())

    def __len__(self) -> int:
        return len(self._store())


class _BraceRewrite(pyc.CustomRewrite):
    """Custom rewrite turning ``macro{ ... }`` into ``macro[ ... ]`` (tuple
    template) or ``macro[__pyc_block__(N)]`` (statement block), and reversing
    either back to ``macro{ ... }`` during ``untransform``.

    The tuple-vs-block decision is per-occurrence and context-sensitive (it parses
    the body), which a static :class:`~pyccolo.AugmentationSpec` can't express --
    hence a custom rewrite rather than a plain paired spec / ``body_func_wrapper``.
    """

    def rewrite(
        self, code: str, register: Callable[[int, int], None]
    ) -> tuple[str, list[tuple[int, int, int]]]:
        names = BraceBlockTracer._macro_names()
        if not names or "{" not in code:
            return code, []
        name_predicate = lambda nm: nm in names  # noqa: E731
        edits: list[tuple[int, int, int]] = []
        bracket_offsets: list[int] = []
        # ``delta`` maps an offset in the *current* (partially-rewritten) ``code``
        # back to the original input: the cumulative length change of all prior
        # (left-of-here) splices. Top-level constructs are matched left-to-right
        # and are disjoint -- a block body is stashed as a string (never
        # re-scanned) and a tuple body is brace-free (a nested ``{`` would make it
        # unparseable as a tuple, routing it to the block path instead) -- so the
        # resulting edits are sorted and non-overlapping in original coordinates.
        delta = 0
        while True:
            match = _find_first_paired_construct(code, name_predicate, "{", "}")
            if match is None:
                break
            starts = _line_starts(code)

            def _abs(pos: tuple[int, int]) -> int:
                return offset_of(starts, pos[0], pos[1])

            name_start = _abs(match.name_start)
            close_end = _abs(match.close_end)
            inner = code[_abs(match.open_end) : _abs(match.close_start)]
            replacement = BraceBlockTracer._emit(match.name, inner)
            edits.append((name_start - delta, close_end - delta, len(replacement)))
            delta += len(replacement) - (close_end - name_start)
            # The ``[`` lands right after NAME; every later splice is to its right,
            # so this offset is already final.
            bracket_offsets.append(name_start + len(match.name))
            code = code[:name_start] + replacement + code[close_end:]
        final_starts = _line_starts(code)
        for off in bracket_offsets:
            anchor = line_col_of(final_starts, off)
            register(anchor.line, anchor.col)
        return code, edits

    def range_for(self, node: ast.AST) -> Range | None:
        # Anchor at the ``[`` (immediately after the macro name), mirroring
        # ``AstRewriter._get_subscript_range_for``. Only Subscript nodes whose
        # ``[`` offset was registered (i.e. brace-derived) actually bind the spec.
        if not isinstance(node, ast.Subscript):
            return None
        end_lineno = getattr(node.value, "end_lineno", None)
        end_col = getattr(node.value, "end_col_offset", None)
        if end_lineno is None or end_col is None:
            return None
        return Range.singleton_span(end_lineno, end_col)

    def reverse(
        self,
        node: ast.AST,
        spec: pyc.AugmentationSpec,
        aug_range: Range,
        code: str,
        line_starts: list[int],
    ) -> tuple[int, int, str] | None:
        from pipescript.tracers.macro_tracer import block_marker_id

        if not isinstance(node, ast.Subscript):
            return None
        open_lineno = getattr(node.value, "end_lineno", None)
        open_col = getattr(node.value, "end_col_offset", None)
        end_lineno = getattr(node, "end_lineno", None)
        end_col = getattr(node, "end_col_offset", None)
        if (
            open_lineno is None
            or open_col is None
            or end_lineno is None
            or end_col is None
        ):
            return None
        open_off = offset_of(line_starts, open_lineno, open_col)
        end_off = offset_of(line_starts, end_lineno, end_col)
        block_id = block_marker_id(node)
        if block_id is not None:
            # Statement block: recover the verbatim source the marker stands for;
            # fall back to the unparsed marker if it's no longer stashed.
            inner = BraceBlockTracer.block_sources.get(block_id)
            if inner is None:
                inner = ast.unparse(node.slice)
        else:
            # Tuple template (``fork{ f1, f2 }``): the slice is the body verbatim.
            sliced = node.slice
            if isinstance(sliced, ast.Index):  # py3.8 compatibility shim
                sliced = sliced.value  # type: ignore[attr-defined]
            if isinstance(sliced, ast.Tuple):
                # Drop the parens ``ast.unparse`` adds around a bare tuple so the
                # resugared body reads ``{f1, f2}`` rather than ``{(f1, f2)}``.
                inner = ", ".join(ast.unparse(e) for e in sliced.elts)
            else:
                inner = ast.unparse(sliced)
        return (open_off, end_off, "{" + inner + "}")


class BraceBlockTracer(pyc.BaseTracer):
    global_guards_enabled = False

    # raw statement-body source keyed by marker id; read by MacroTracer. Backed
    # by per-thread storage (see ``_ThreadLocalDict``) so a lint/analysis pass on
    # a background thread can't clobber the body a concurrent execution registered.
    block_sources: MutableMapping[int, str] = _ThreadLocalDict()
    # reverse map (source -> id) so a given block always gets the *same* marker:
    # ipyflow invokes the syntax augmenter several times per cell (liveness,
    # analysis, execution), and a fresh id each pass would make the augmenter
    # non-idempotent -- the rewriter's instrumentation, set up against one pass's
    # output, then fails to line up with the marker the executed pass emits, and
    # the cell silently runs uninstrumented. (Those passes share a thread, so the
    # per-thread store keeps dedup idempotent within a cell.)
    _id_by_source: MutableMapping[str, int] = _ThreadLocalDict()
    _counter = 0

    # The brace rewrite, as a custom augmentation so it threads positions and is
    # reversible via ``untransform``. This is BraceBlockTracer's only spec, so it
    # runs in this (outermost) tracer's augmentation pass, before PipelineTracer's.
    brace_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.custom,
        token="{",
        replacement="[",
        custom=_BraceRewrite(),
    )

    @staticmethod
    def _macro_names() -> set[str]:
        from pipescript.tracers.macro_tracer import MacroTracer

        return (
            set(MacroTracer.static_macros)
            | set(MacroTracer.dynamic_macros)
            | set(MacroTracer.dynamic_method_macros)
        )

    @staticmethod
    def _is_tuple_expression(inner: str) -> bool:
        from pipescript.tracers.pipeline_tracer import PipelineTracer

        # A top-level tuple body is the multi-function template form consumed by
        # `fork`/`parallel` (e.g. `fork{ f1, f2 }`), which genuinely needs the
        # expression/template path. Everything else -- single expressions
        # included -- goes through the block path so braces uniformly use the
        # block's collapse-`$` semantics (and get their own scope, rather than
        # the quick-lambda placeholder logic which mis-scopes when nested).
        check = PipelineTracer.instance().preprocess(inner, None)
        try:
            tree = ast.parse(check.strip(), mode="eval")
        except SyntaxError:
            return False
        return isinstance(tree.body, ast.Tuple)

    @classmethod
    def _emit(cls, name: str, inner: str) -> str:
        if cls._is_tuple_expression(inner):
            # fork/parallel multi-function template: brace-for-bracket swap and
            # let the normal expression machinery handle it. (Side-effect free, so
            # it's safe under a pure transform too.)
            return f"{name}[{inner}]"
        # Emit the marker as a call to the defined sentinel `__pyc_block__(N)`
        # rather than a bare (undefined) name: under ipyflow the slice expression
        # is evaluated before the macro handler can substitute it, so an
        # undefined marker name would leak as a `NameError`. See
        # `macro_tracer.block_marker_id` / `_block_marker_sentinel`.
        from pipescript.tracers.macro_tracer import BLOCK_MARKER_FUNC

        if pyc.is_pure_transform():
            # Analysis-only transform (lint / format / source-map): emit a valid,
            # lintable marker but DON'T register a body or bump shared counters.
            # The lowered code is never executed, and mutating the process-global
            # registries the runtime later reads would corrupt a concurrent
            # execution (the original bug this guards against).
            return f"{name}[{BLOCK_MARKER_FUNC}(0)]"
        # Everything else (single expressions and statement bodies) is stashed
        # and replaced with a marker. The marker flows through macro expansion
        # (e.g. `foreach` expands to `... |> map[do[<marker>]] |> ...`) and is
        # compiled into a function -- with collapse-`$` semantics and its own
        # scope -- by MacroTracer when the consuming macro is expanded.
        n = cls._id_by_source.get(inner)
        if n is None:
            cls._counter += 1
            n = cls._counter
            cls._id_by_source[inner] = n
            cls.block_sources[n] = inner
        return f"{name}[{BLOCK_MARKER_FUNC}({n})]"
