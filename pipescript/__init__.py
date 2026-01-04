"""
pipescript: powerful pipeline syntax for IPython and Jupyter.
Just run `%load_ext pipescript` to begin using pipe operators, placeholders, and more.
"""

from __future__ import annotations

from IPython.core.interactiveshell import InteractiveShell

import pipescript.api
from pipescript.api import *  # noqa: F403
from pipescript.patches.completion_patch import patch_completer, unpatch_completer

from . import _version  # noqa: E402

__version__ = _version.get_versions()["version"]


def clear_tracer_stacks(*_, **__) -> None:
    from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer
    from pipescript.tracers.pipeline_tracer import PipelineTracer

    OptionalChainingTracer.instance().clear_stacks()
    PipelineTracer.instance().clear_stacks()


def load_ipython_extension(shell: InteractiveShell) -> None:
    from ipyflow.shell.interactiveshell import IPyflowInteractiveShell

    from pipescript.tracers.macro_tracer import MacroTracer
    from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer
    from pipescript.tracers.pipeline_tracer import PipelineTracer

    if not isinstance(shell, IPyflowInteractiveShell):
        shell.run_line_magic("load_ext", "ipyflow.shell")
        shell.run_line_magic("flow", "deregister all")
    assert isinstance(shell, IPyflowInteractiveShell)
    shell.run_line_magic(
        "flow", f"register {PipelineTracer.__module__}.{PipelineTracer.__name__}"
    )
    shell.run_line_magic(
        "flow",
        f"register {MacroTracer.__module__}.{MacroTracer.__name__}",
    )
    shell.run_line_magic(
        "flow",
        f"register {OptionalChainingTracer.__module__}.{OptionalChainingTracer.__name__}",
    )
    shell.events.register("post_run_cell", clear_tracer_stacks)
    patch_completer(shell.Completer)


def unload_ipython_extension(shell: InteractiveShell) -> None:
    from pipescript.tracers.macro_tracer import MacroTracer
    from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer
    from pipescript.tracers.pipeline_tracer import PipelineTracer

    unpatch_completer(shell.Completer)
    shell.events.unregister("post_run_cell", clear_tracer_stacks)
    shell.run_line_magic(
        "flow",
        f"deregister {OptionalChainingTracer.__module__}.{OptionalChainingTracer.__name__}",
    )
    shell.run_line_magic(
        "flow",
        f"deregister {MacroTracer.__module__}.{MacroTracer.__name__}",
    )
    shell.run_line_magic(
        "flow", f"deregister {PipelineTracer.__module__}.{PipelineTracer.__name__}"
    )


__all__ = list(pipescript.api.__all__)
