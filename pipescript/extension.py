from __future__ import annotations

import functools
import os
import re
import sys
from types import TracebackType
from typing import TYPE_CHECKING, Callable, cast

import pyccolo as pyc
from pyccolo.tracer import PYCCOLO_DEV_MODE_ENV_VAR

from pipescript.patches.completion_patch import patch_completer, unpatch_completer
from pipescript.tracers.brace_block_tracer import BraceBlockTracer
from pipescript.tracers.macro_tracer import DynamicMacro, MacroTracer
from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer
from pipescript.tracers.pipeline_tracer import PipelineTracer

if TYPE_CHECKING:
    from IPython.core.interactiveshell import InteractiveShell


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


# pipescript's tracers, in the order their syntax augmenters must run.
# ``BraceBlockTracer`` is first so `macro{ ... }` brace extraction happens before
# the `$` -> `_` placeholder pass. pyccolo registers in this order and keeps it,
# whether the host is plain IPython or ipyflow.
PIPESCRIPT_TRACERS: list[type[pyc.BaseTracer]] = [
    BraceBlockTracer,
    PipelineTracer,
    MacroTracer,
    OptionalChainingTracer,
]


def register_tracers(shell: InteractiveShell) -> list[pyc.BaseTracer]:
    for tracer_cls in PIPESCRIPT_TRACERS:
        pyc.register_ipython_tracer(tracer_cls, shell=shell)
    return [cast(pyc.BaseTracer, cls).instance() for cls in PIPESCRIPT_TRACERS]


def deregister_tracers(shell: InteractiveShell) -> None:
    for tracer_cls in reversed(PIPESCRIPT_TRACERS):
        pyc.deregister_ipython_tracer(tracer_cls, shell=shell)


# pyccolo owns the frame filter now; both hosts share the one implementation.
filter_hidden_frames = pyc.filter_hidden_frames


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


def load_ipython_extension(
    shell: InteractiveShell,
    run_cell: Callable[[str], object] | None = None,
) -> None:
    """Install pipescript's tracers on whatever host owns the cell lifecycle.

    pyccolo's IPython extension owns the AST/input transformers, the cell
    filename, and the per-cell tracing context -- and it already knows whether
    ipyflow is driving. So there is one code path here, not two.
    """
    # pyccolo's extension owns the cell tracing driver; loading it through the
    # extension manager (rather than calling ``load_ipython_extension`` directly)
    # keeps a later ``%unload_ext pyccolo`` working.
    assert shell.extension_manager is not None
    shell.extension_manager.load_extension("pyccolo")
    tracers = register_tracers(shell)
    shell.events.register("post_run_cell", clear_tracer_stacks)
    shell.events.register("post_run_cell", identify_dynamic_macros)
    # monkey patch instead of using set_custom_exc so that
    # we don't interfere with other callers of set_custom_exc
    shell.__class__.showtraceback = make_patched_showtraceback(  # type: ignore[method-assign]
        shell.__class__.showtraceback
    )
    patch_completer(shell, tracers)

    if run_cell is None:
        load_builtin_dynamic_macros(shell)
        return

    # A caller-supplied ``run_cell`` (ipyflow's test harness) needs the macro
    # definitions deferred until after the first cell, so the just-registered
    # tracers are live when the definitions are evaluated.
    def _load_builtin_dynamic_macros_once(*_args, **_kwargs) -> None:
        shell.events.unregister("post_run_cell", _load_builtin_dynamic_macros_once)
        load_builtin_dynamic_macros(shell, run_cell=run_cell)

    shell.events.register("post_run_cell", _load_builtin_dynamic_macros_once)


def unload_ipython_extension(shell: InteractiveShell) -> None:
    unpatch_completer(shell)
    for handler in (identify_dynamic_macros, clear_tracer_stacks):
        try:
            shell.events.unregister("post_run_cell", handler)
        except ValueError:
            pass
    deregister_tracers(shell)
