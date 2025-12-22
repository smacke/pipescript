# -*- coding: utf-8 -*-
"""
nbpipes: powerful pipeline syntax for IPython and Jupyter.
Just run `%load_ext nbpipes` to begin using pipe operators, placeholders, and more.
"""
from __future__ import annotations
from IPython.core.interactiveshell import InteractiveShell

from . import _version  # noqa: E402
__version__ = _version.get_versions()['version']


def load_ipython_extension(shell: InteractiveShell) -> None:
	from ipyflow.shell.interactiveshell import IPyflowInteractiveShell

	if not isinstance(shell, IPyflowInteractiveShell):
		shell.run_line_magic("load_ext", "ipyflow.shell")
	assert isinstance(shell, IPyflowInteractiveShell)
	shell.run_line_magic("flow", "deregister dataflow")
	shell.run_line_magic("flow", "register pyccolo.examples.PipelineTracer")
	shell.run_line_magic("flow", "register pyccolo.examples.QuickLambdaTracer")

