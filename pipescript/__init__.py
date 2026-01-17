"""
pipescript: powerful pipeline syntax for IPython and Jupyter.
Just run `%load_ext pipescript` to begin using pipe operators, placeholders, and more.
"""

from __future__ import annotations

import sys

from IPython.core.interactiveshell import InteractiveShell

import pipescript.api
from pipescript.api import *  # noqa: F403
from pipescript.extension import clear_tracer_stacks, identify_dynamic_macros
from pipescript.extension import load_ipython_extension as load_ipython_extension_base
from pipescript.extension import (
    unload_ipython_extension as unload_ipython_extension_base,
)
from pipescript.patches.completion_patch import patch_completer, unpatch_completer
from pipescript.tracers.macro_tracer import MacroTracer
from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer
from pipescript.tracers.pipeline_tracer import PipelineTracer

from . import _version  # noqa: E402

__version__ = _version.get_versions()["version"]


def load_ipython_extension_ipyflow(shell: InteractiveShell) -> None:
    from ipyflow.shell.interactiveshell import IPyflowInteractiveShell

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
    shell.events.register("post_run_cell", identify_dynamic_macros)
    patch_completer(shell.Completer)


def unload_ipython_extension_ipyflow(shell: InteractiveShell) -> None:
    unpatch_completer(shell.Completer)
    shell.events.unregister("post_run_cell", identify_dynamic_macros)
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


def load_ipython_extension(shell: InteractiveShell) -> None:
    IPyflowInteractiveShell = getattr(
        sys.modules.get("ipyflow.shell.interactiveshell"),
        "IPyflowInteractiveShell",
        type(None),
    )
    if isinstance(shell, IPyflowInteractiveShell):
        load_ipython_extension_ipyflow(shell)
    else:
        load_ipython_extension_base(shell)


def unload_ipython_extension(shell: InteractiveShell) -> None:
    IPyflowInteractiveShell = getattr(
        sys.modules.get("ipyflow.shell.interactiveshell"),
        "IPyflowInteractiveShell",
        type(None),
    )
    if isinstance(shell, IPyflowInteractiveShell):
        unload_ipython_extension_ipyflow(shell)
    else:
        unload_ipython_extension_base(shell)


__all__ = list(pipescript.api.__all__)
