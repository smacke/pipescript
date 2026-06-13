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

Enter this tracer **outermost** (before ``PipelineTracer``) so the brace
extraction happens before the ``$`` -> ``_`` placeholder pass; that keeps the
captured body's ``$`` intact (re-processed later) and avoids registering stale
placeholder positions for source that has been moved out of band.
"""

from __future__ import annotations

import ast

import pyccolo as pyc
from pyccolo.syntax_augmentation import make_paired_delimiter_augmenter


class BraceBlockTracer(pyc.BaseTracer):
    global_guards_enabled = False

    # raw statement-body source keyed by marker id; read by MacroTracer
    block_sources: dict[int, str] = {}
    # reverse map (source -> id) so a given block always gets the *same* marker:
    # ipyflow invokes the syntax augmenter several times per cell (liveness,
    # analysis, execution), and a fresh id each pass would make the augmenter
    # non-idempotent -- the rewriter's instrumentation, set up against one pass's
    # output, then fails to line up with the marker the executed pass emits, and
    # the cell silently runs uninstrumented.
    _id_by_source: dict[str, int] = {}
    _counter = 0

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

    def _emit(self, name: str, inner: str) -> str:
        if self._is_tuple_expression(inner):
            # fork/parallel multi-function template: brace-for-bracket swap and
            # let the normal expression machinery handle it.
            return f"{name}[{inner}]"
        # Everything else (single expressions and statement bodies) is stashed
        # and replaced with a marker. The marker flows through macro expansion
        # (e.g. `foreach` expands to `... |> map[do[<marker>]] |> ...`) and is
        # compiled into a function -- with collapse-`$` semantics and its own
        # scope -- by MacroTracer when the consuming macro is expanded.
        n = BraceBlockTracer._id_by_source.get(inner)
        if n is None:
            BraceBlockTracer._counter += 1
            n = BraceBlockTracer._counter
            BraceBlockTracer._id_by_source[inner] = n
            BraceBlockTracer.block_sources[n] = inner
        # Emit the marker as a call to the defined sentinel `__pyc_block__(N)`
        # rather than a bare (undefined) name: under ipyflow the slice expression
        # is evaluated before the macro handler can substitute it, so an
        # undefined marker name would leak as a `NameError`. See
        # `macro_tracer.block_marker_id` / `_block_marker_sentinel`.
        from pipescript.tracers.macro_tracer import BLOCK_MARKER_FUNC

        return f"{name}[{BLOCK_MARKER_FUNC}({n})]"

    def _augment(self, code: str) -> str:
        names = self._macro_names()
        if not names:
            return code
        return make_paired_delimiter_augmenter(names, self._emit)(code)

    def preprocess(self, code, rewriter):
        code = super().preprocess(code, rewriter)
        return self._augment(code)

    def make_syntax_augmenter(self, ast_rewriter):
        base = super().make_syntax_augmenter(ast_rewriter)

        def _aug(lines):
            out = base(lines)
            if isinstance(out, list):
                return self._augment("".join(out)).splitlines(keepends=True)
            return self._augment(out)

        return _aug
