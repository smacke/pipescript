"""
nbpipes: powerful pipeline syntax for IPython and Jupyter.
Just run `%load_ext nbpipes` to begin using pipe operators, placeholders, and more.
"""

from __future__ import annotations

from IPython.core.interactiveshell import InteractiveShell

from . import _version  # noqa: E402
from .completion_patch import patch_completer

__version__ = _version.get_versions()["version"]


def load_ipython_extension(shell: InteractiveShell) -> None:
    from ipyflow.shell.interactiveshell import IPyflowInteractiveShell

    from nbpipes.macro_tracer import MacroTracer
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
    patch_completer(shell.Completer)
