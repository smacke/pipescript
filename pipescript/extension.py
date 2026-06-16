from __future__ import annotations

import functools
import inspect
import os
import re
import sys
from types import FrameType, TracebackType
from typing import TYPE_CHECKING, Any, Callable, cast

import pyccolo as pyc
from pyccolo.emit_event import SANDBOX_FNAME_PREFIX
from pyccolo.tracer import (
    HIDE_PYCCOLO_FRAME,
    PYCCOLO_DEV_MODE_ENV_VAR,
    TRACED_LAMBDA_NAME,
)

from pipescript.patches.completion_patch import patch_completer
from pipescript.tracers.brace_block_tracer import BraceBlockTracer
from pipescript.tracers.macro_tracer import DynamicMacro, MacroTracer
from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer
from pipescript.tracers.pipeline_tracer import PipelineTracer

if TYPE_CHECKING:
    from IPython.core.interactiveshell import ExecutionInfo, InteractiveShell


def clear_tracer_stacks(*_, **__) -> None:
    from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer
    from pipescript.tracers.pipeline_tracer import PipelineTracer

    OptionalChainingTracer.instance().clear_stacks()
    PipelineTracer.instance().clear_stacks()


def identify_dynamic_macros(*_, **__) -> None:
    from IPython import get_ipython

    shell = get_ipython()
    if shell is None:
        return
    user_ns = shell.user_ns
    MacroTracer.dynamic_macros.clear()
    MacroTracer.dynamic_method_macros.clear()
    for k, v in user_ns.items():
        if not isinstance(v, DynamicMacro):
            continue
        if v.is_method:
            MacroTracer.dynamic_method_macros[k] = v
        else:
            MacroTracer.dynamic_macros[k] = v


def make_ipython_name(
    shell: InteractiveShell,
    info: ExecutionInfo,
) -> str:
    cache = shell.compile.cache
    kwargs: dict[str, Any] = {}
    if "raw_code" in inspect.signature(cache).parameters:
        kwargs["raw_code"] = info.raw_cell
    transformed_cell = getattr(info, "transformed_cell", None)
    return cache(
        transformed_cell or shell.transform_cell(info.raw_cell),
        shell.execution_count,
        **kwargs,
    )


def make_tracing_contexts(
    shell: InteractiveShell, tracers: list[pyc.BaseTracer]
) -> tuple[Callable[..., None], Callable[..., None]]:
    inited = False
    rewriter = tracers[-1].ast_rewriter_cls(tracers, "")

    def _cleanup_transformer(lines) -> None:
        rewriter.__init__(tracers, "")  # type: ignore[misc]
        return lines

    shell.input_transformers_cleanup.insert(0, _cleanup_transformer)

    for tracer in tracers:
        shell.input_transformers_post.append(tracer.make_syntax_augmenter(rewriter))
    shell.ast_transformers.append(rewriter)

    def _enter_tracer_context_callback(info: ExecutionInfo, *_, **__) -> None:
        for tracer in tracers:
            tracer.reset()
            tracer.__enter__()
        cell_name = make_ipython_name(shell, info)
        rewriter._path = cell_name
        rewriter._module_id = shell.execution_count
        for tracer in tracers:
            tracer._tracing_enabled_files.add(cell_name)

    def _exit_tracer_context_callback(*_, **__) -> None:
        nonlocal inited
        was_inited = inited
        inited = True
        if not was_inited:
            return
        for tracer in reversed(tracers):
            tracer.__exit__(None, None, None)

    return _enter_tracer_context_callback, _exit_tracer_context_callback


def filter_hidden_frames(tb: TracebackType | None) -> None:
    prev = None
    while tb is not None:
        should_filter = False
        frame: FrameType = tb.tb_frame
        if prev is not None:
            fname = frame.f_code.co_filename
            # A sandbox frame marked traceback-visible (a compiled pipescript
            # block, or a macro sub-lambda like a `fork` branch) is
            # user-meaningful and pinpoints the failing stage -- keep it instead
            # of filtering it as a synthetic sandbox frame.
            should_filter = frame.f_locals.get(HIDE_PYCCOLO_FRAME, False)
            should_filter = should_filter or (
                fname.startswith(SANDBOX_FNAME_PREFIX)
                and not pyc.is_traceback_visible(fname)
            )
            should_filter = should_filter or frame.f_code.co_name in (
                TRACED_LAMBDA_NAME,
                "_patched_eval",
                "_patched_tracer_eval",
            )
            should_filter = should_filter or "pyccolo" in fname
        if should_filter and prev is not None:
            prev.tb_next = tb.tb_next
        else:
            prev = tb
        tb = tb.tb_next


_BLOCK_MARKER_RE = re.compile(r"\[__pyc_block__\(\d+\)\]")


def resugar_block_markers(tb: TracebackType | None) -> None:
    """Rewrite ``map[__pyc_block__(N)]`` markers to ``map{...}`` in the displayed
    source of any frame in ``tb`` (most visibly the cell line), so the user sees
    the brace block they wrote rather than the desugared marker."""
    import linecache

    seen: set[str] = set()
    while tb is not None:
        fname = tb.tb_frame.f_code.co_filename
        if fname not in seen:
            seen.add(fname)
            entry = linecache.cache.get(fname)
            if entry is not None and len(entry) == 4:
                size, mtime, lines, fullname = entry
                if lines and "__pyc_block__" in "".join(lines):
                    linecache.cache[fname] = (
                        size,
                        mtime,
                        [_BLOCK_MARKER_RE.sub("{...}", ln) for ln in lines],
                        fullname,
                    )
        tb = tb.tb_next


def make_patched_showtraceback(orig_showtraceback):
    from pipescript.patches.diagnostics import annotate_pipescript_exception

    @functools.wraps(orig_showtraceback)
    def patched_showtraceback(self, *args, **kwargs):
        evalue = None
        if os.getenv(PYCCOLO_DEV_MODE_ENV_VAR) != "1":
            etype, evalue, tb = self._get_exc_info(kwargs.get("exc_tuple"))
            filter_hidden_frames(tb)
            try:
                resugar_block_markers(tb)
                annotate_pipescript_exception(etype, evalue, tb)
            except Exception:
                pass
        orig_showtraceback(self, *args, **kwargs)
        # On <3.11 IPython won't render exception notes, so surface the
        # pipescript ones ourselves (3.11+ shows native __notes__ already).
        if sys.version_info < (3, 11) and evalue is not None:
            notes = getattr(evalue, "_pyc_notes", None)
            if notes:
                sys.stderr.write("\n".join(notes) + "\n")

    return patched_showtraceback


def load_builtin_dynamic_macros(
    shell: InteractiveShell,
    run_cell: Callable[[str], object] | None = None,
) -> None:
    if run_cell is None:

        def run_cell(code: str) -> object:
            return shell.run_cell(code, store_history=False, silent=False)

    for macro_name, macro_def in MacroTracer.builtin_dynamic_macro_definitions.items():
        run_cell(f"{macro_name} = {macro_def}")


def load_ipython_extension(shell: InteractiveShell) -> None:
    tracers = [
        cast(pyc.BaseTracer, cls).instance()
        # BraceBlockTracer must be outermost so `macro{ ... }` brace extraction
        # runs before the `$` -> `_` placeholder pass.
        for cls in [
            BraceBlockTracer,
            PipelineTracer,
            MacroTracer,
            OptionalChainingTracer,
        ]
    ]
    enter_context, exit_context = make_tracing_contexts(shell, tracers)
    shell.events.register("pre_run_cell", enter_context)
    shell.events.register("post_run_cell", exit_context)
    shell.events.register("post_run_cell", clear_tracer_stacks)
    shell.events.register("post_run_cell", identify_dynamic_macros)
    # monkey patch instead of using set_custom_exc so that
    # we don't interfere with other callers of set_custom_exc
    shell.__class__.showtraceback = make_patched_showtraceback(  # type: ignore[method-assign]
        shell.__class__.showtraceback
    )
    patch_completer(shell, tracers)
    load_builtin_dynamic_macros(shell)


def unload_ipython_extension(_shell: InteractiveShell) -> None:
    # TODO: implement this
    pass
