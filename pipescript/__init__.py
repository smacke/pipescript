"""
pipescript: powerful pipeline syntax for IPython and Jupyter.
Just run `%load_ext pipescript` to begin using pipe operators, placeholders, and more.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import pipescript.api
from pipescript.api import *  # noqa: F403
from pipescript.extension import (  # noqa: F401
    PIPESCRIPT_TRACERS,
    clear_tracer_stacks,
    identify_dynamic_macros,
    load_builtin_dynamic_macros,
)
from pipescript.extension import load_ipython_extension as load_ipython_extension_base
from pipescript.extension import (
    unload_ipython_extension as unload_ipython_extension_base,
)
from pipescript.tracers.brace_block_tracer import BraceBlockTracer  # noqa: F401
from pipescript.tracers.macro_tracer import MacroTracer  # noqa: F401
from pipescript.tracers.optional_chaining_tracer import (  # noqa: F401
    OptionalChainingTracer,
)
from pipescript.tracers.pipeline_tracer import PipelineTracer  # noqa: F401

from . import _version  # noqa: E402

__version__ = _version.get_versions()["version"]

if TYPE_CHECKING:
    from IPython.core.interactiveshell import InteractiveShell


def load_ipython_extension(
    shell: InteractiveShell,
    run_cell: Callable[[str], object] | None = None,
) -> None:
    load_ipython_extension_base(shell, run_cell=run_cell)


def unload_ipython_extension(shell: InteractiveShell) -> None:
    unload_ipython_extension_base(shell)


# Back-compat aliases. Host detection now lives in pyccolo's IPython extension,
# so both hosts take the same path and these are no longer distinct.
load_ipython_extension_ipyflow = load_ipython_extension
unload_ipython_extension_ipyflow = unload_ipython_extension


__all__ = list(pipescript.api.__all__)
