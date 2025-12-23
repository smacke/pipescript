# -*- coding: utf-8 -*-
"""
nbpipes: powerful pipeline syntax for IPython and Jupyter.
Just run `%load_ext nbpipes` to begin using pipe operators, placeholders, and more.
"""
from __future__ import annotations

from IPython.core.completer import Completer, Completion
from IPython.core.interactiveshell import InteractiveShell

from . import _version  # noqa: E402
__version__ = _version.get_versions()['version']


def _patch_completer(completer: Completer) -> None:
	class PatchedCompleter(completer.__class__):
		def completions(self, text: str, offset: int) -> list[Completion]:
			before_offset = text[:offset]
			placeholder_split = before_offset.rsplit("$.", 1)
			if len(placeholder_split) == 2:
				shift_amount = len(placeholder_split[0])
				completions = super().completions(
					"_." + placeholder_split[1], offset - shift_amount)
				completions = list(completions)
				for completion in completions:
					completion.start += shift_amount
					completion.end += shift_amount
				return completions
			else:
				return super().completions(text, offset)

	completer.__class__ = PatchedCompleter


def load_ipython_extension(shell: InteractiveShell) -> None:
	from ipyflow.shell.interactiveshell import IPyflowInteractiveShell

	if not isinstance(shell, IPyflowInteractiveShell):
		shell.run_line_magic("load_ext", "ipyflow.shell")
		shell.run_line_magic("flow", "deregister all")
	assert isinstance(shell, IPyflowInteractiveShell)
	shell.run_line_magic("flow", "register pyccolo.examples.PipelineTracer")
	shell.run_line_magic("flow", "register pyccolo.examples.QuickLambdaTracer")
	_patch_completer(shell.Completer)

