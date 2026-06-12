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
    def _is_expression(inner: str) -> bool:
        from pipescript.tracers.pipeline_tracer import PipelineTracer

        # apply the pipeline/placeholder augmentations so that $, |>, etc. become
        # valid Python, then ask whether the body parses as a single expression.
        check = PipelineTracer.instance().preprocess(inner, None)
        try:
            ast.parse(check.strip(), mode="eval")
            return True
        except SyntaxError:
            return False

    def _emit(self, name: str, inner: str) -> str:
        if self._is_expression(inner):
            # expression bodies are just a brace-for-bracket swap; the normal
            # machinery (incl. the `$` placeholder pass) takes it from here.
            return f"{name}[{inner}]"
        # statement body: stash it and leave a marker. The marker flows through
        # macro expansion (e.g. a dynamic method macro like `foreach` expands to
        # `... |> map[do[<marker>]] |> ...`) and is turned into a function by
        # MacroTracer when the (static) macro that consumes it is expanded.
        BraceBlockTracer._counter += 1
        n = BraceBlockTracer._counter
        BraceBlockTracer.block_sources[n] = inner
        return f"{name}[__pyc_block_{n}__]"

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
