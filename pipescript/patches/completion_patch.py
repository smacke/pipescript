from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import pyccolo as pyc

if TYPE_CHECKING:
    from ipykernel.ipkernel import IPythonKernel
    from IPython.core.completer import Completer, Completion
    from IPython.core.interactiveshell import InteractiveShell


orig_do_complete = None
orig_get_completion_context = None


def patch_kernel_completer(
    kernel: IPythonKernel, tracers: list[pyc.BaseTracer]
) -> None:
    global orig_do_complete
    global orig_get_completion_context
    from ipykernel.ipkernel import IPythonKernel

    orig_do_complete = IPythonKernel.do_complete

    @functools.wraps(IPythonKernel.do_complete)
    def patched_do_complete(self, code: str, cursor_pos: int) -> dict[str, Any]:
        before_offset = code[:cursor_pos]
        transformed = pyc.transform(before_offset, tracers=tracers)
        if transformed == before_offset:
            return orig_do_complete(self, code, cursor_pos)
        completions = orig_do_complete(self, transformed, len(transformed))
        shift_amount = cursor_pos - len(transformed)
        completions["cursor_start"] += shift_amount
        completions["cursor_end"] += shift_amount
        for exp in completions.get("metadata", {}).get(
            "_jupyter_types_experimental", []
        ):
            exp["start"] += shift_amount
            exp["end"] += shift_amount
        return completions

    IPythonKernel.do_complete = patched_do_complete  # type: ignore[method-assign]

    # Databricks LSP support
    if hasattr(kernel.__class__, "_get_completion_context"):
        orig_get_completion_context = kernel.__class__._get_completion_context

        @functools.wraps(kernel.__class__._get_completion_context)
        def patched_get_completion_context(self, *args, **kwargs) -> str:
            return pyc.transform(
                orig_get_completion_context(self, *args, **kwargs), tracers=tracers
            )

        kernel.__class__._get_completion_context = patched_get_completion_context


def unpatch_kernel_completer(kernel: IPythonKernel) -> None:
    global orig_do_complete
    global orig_get_completion_context
    assert orig_do_complete is not None
    from ipykernel.ipkernel import IPythonKernel

    IPythonKernel.do_complete = orig_do_complete  # type: ignore[method-assign]
    orig_do_complete = None

    if (
        hasattr(kernel.__class__, "_get_completion_context")
        and orig_get_completion_context is not None
    ):
        delattr(kernel.__class__, "_get_completion_context")
        orig_get_completion_context = None


def patch_shell_completer(completer: Completer, tracers: list[pyc.BaseTracer]) -> None:
    clazz: type[Completer] = completer.__class__

    class PatchedCompleter(clazz):  # type: ignore[misc, valid-type]
        def completions(self, text: str, offset: int) -> list[Completion]:
            before_offset = text[:offset]
            transformed = pyc.transform(before_offset, tracers=tracers)
            if transformed == before_offset:
                return super().completions(text, offset)
            completions = list(super().completions(transformed, len(transformed)))
            shift_amount = offset - len(transformed)
            for completion in completions:
                completion.start += shift_amount
                completion.end += shift_amount
            return completions

    completer.__class__ = PatchedCompleter


def unpatch_shell_completer(completer: Completer) -> None:
    completer.__class__ = completer.__class__.mro()[1]


def patch_completer(shell: InteractiveShell, tracers: list[pyc.BaseTracer]) -> None:
    if (kernel := getattr(shell, "kernel", None)) is None:
        patch_shell_completer(shell.Completer, tracers)
    else:
        patch_kernel_completer(kernel, tracers)


def unpatch_completer(shell: InteractiveShell) -> None:
    if (kernel := getattr(shell, "kernel", None)) is None:
        unpatch_shell_completer(shell.Completer)
    else:
        unpatch_kernel_completer(kernel)
