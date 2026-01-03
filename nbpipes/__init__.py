"""
nbpipes: powerful pipeline syntax for IPython and Jupyter.
Just run `%load_ext nbpipes` to begin using pipe operators, placeholders, and more.
"""

from __future__ import annotations

from IPython.core.interactiveshell import InteractiveShell

from . import _version  # noqa: E402
from .completion_patch import patch_completer, unpatch_completer

__version__ = _version.get_versions()["version"]


def clear_tracer_stacks(*_, **__) -> None:
    from nbpipes.nullish_tracer import NullishTracer
    from nbpipes.pipeline_tracer import PipelineTracer

    NullishTracer.instance().clear_stacks()
    PipelineTracer.instance().clear_stacks()


def load_ipython_extension(shell: InteractiveShell) -> None:
    from ipyflow.shell.interactiveshell import IPyflowInteractiveShell

    from nbpipes.macro_tracer import MacroTracer
    from nbpipes.nullish_tracer import NullishTracer
    from nbpipes.pipeline_tracer import PipelineTracer

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
        f"register {NullishTracer.__module__}.{NullishTracer.__name__}",
    )
    shell.events.register("post_run_cell", clear_tracer_stacks)
    patch_completer(shell.Completer)


def unload_ipython_extension(shell: InteractiveShell) -> None:
    from nbpipes.macro_tracer import MacroTracer
    from nbpipes.nullish_tracer import NullishTracer
    from nbpipes.pipeline_tracer import PipelineTracer

    unpatch_completer(shell.Completer)
    shell.events.unregister("post_run_cell", clear_tracer_stacks)
    shell.run_line_magic(
        "flow",
        f"deregister {NullishTracer.__module__}.{NullishTracer.__name__}",
    )
    shell.run_line_magic(
        "flow",
        f"deregister {MacroTracer.__module__}.{MacroTracer.__name__}",
    )
    shell.run_line_magic(
        "flow", f"deregister {PipelineTracer.__module__}.{PipelineTracer.__name__}"
    )
