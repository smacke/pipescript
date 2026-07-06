from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import pyccolo as pyc

if TYPE_CHECKING:
    from ipykernel.ipkernel import IPythonKernel
    from IPython.core.completer import Completer, Completion
    from IPython.core.interactiveshell import InteractiveShell


do_complete_patch_cls: type[IPythonKernel] = None  # type: ignore[assignment]
get_completion_context_patch_cls: type[IPythonKernel] = None  # type: ignore[assignment]
orig_do_complete = None
orig_get_completion_context = None
_did_patch_shell_completer = False


def patch_kernel_completer(
    kernel: IPythonKernel, tracers: list[pyc.BaseTracer]
) -> bool:
    global do_complete_patch_cls
    global get_completion_context_patch_cls
    global orig_do_complete
    global orig_get_completion_context

    patch_cls = None
    for cls in kernel.__class__.mro():
        if "do_complete" in cls.__dict__:
            patch_cls = cls
            break
    if patch_cls is None:
        # Some kernels (e.g. the JupyterLite/Pyodide `PyodideKernel`) expose no
        # `do_complete` method to wrap; signal the caller to fall back to
        # patching the shell completer instead.
        return False
    do_complete_patch_cls = patch_cls
    orig_do_complete = do_complete_patch_cls.do_complete

    @functools.wraps(do_complete_patch_cls.do_complete)
    def patched_do_complete(self, code: str, cursor_pos: int) -> dict[str, Any]:
        before_offset = code[:cursor_pos]
        transformed = pyc.transform(before_offset, tracers=tracers)
        if transformed == before_offset:
            return orig_do_complete(self, code, cursor_pos)
        completions = orig_do_complete(self, transformed, len(transformed))
        shift_amount = cursor_pos - len(transformed)
        completions["cursor_start"] += shift_amount
        completions["cursor_end"] += shift_amount
        for metadatum in completions.get("metadata", {}).get(
            "_jupyter_types_experimental", []
        ):
            metadatum["start"] += shift_amount
            metadatum["end"] += shift_amount
        return completions

    do_complete_patch_cls.do_complete = patched_do_complete  # type: ignore[method-assign]

    # Databricks LSP support
    if not hasattr(kernel.__class__, "_get_completion_context"):
        return True
    for get_completion_context_patch_cls in kernel.__class__.mro():
        if "_get_completion_context" in get_completion_context_patch_cls.__dict__:
            break
    orig_get_completion_context = (
        get_completion_context_patch_cls._get_completion_context  # type: ignore[attr-defined]
    )

    @functools.wraps(get_completion_context_patch_cls._get_completion_context)  # type: ignore[attr-defined]
    def patched_get_completion_context(self, *args, **kwargs) -> str:
        return pyc.transform(
            orig_get_completion_context(self, *args, **kwargs), tracers=tracers
        )

    get_completion_context_patch_cls._get_completion_context = (  # type: ignore[attr-defined]
        patched_get_completion_context
    )
    return True


def unpatch_kernel_completer() -> None:
    global do_complete_patch_cls
    global get_completion_context_patch_cls
    global orig_do_complete
    global orig_get_completion_context

    assert do_complete_patch_cls is not None
    assert orig_do_complete is not None
    do_complete_patch_cls.do_complete = orig_do_complete  # type: ignore[method-assign]
    do_complete_patch_cls = None  # type: ignore[assignment]
    orig_do_complete = None

    if get_completion_context_patch_cls is None:
        return
    assert orig_get_completion_context is not None
    get_completion_context_patch_cls._get_completion_context = orig_get_completion_context  # type: ignore[attr-defined]
    get_completion_context_patch_cls = None  # type: ignore[assignment]


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
    global _did_patch_shell_completer
    kernel = getattr(shell, "kernel", None)
    if kernel is None or not patch_kernel_completer(kernel, tracers):
        patch_shell_completer(shell.Completer, tracers)
        _did_patch_shell_completer = True
    else:
        _did_patch_shell_completer = False


def unpatch_completer(shell: InteractiveShell) -> None:
    # Mirror the branch taken in patch_completer: under a Pyodide kernel
    # `shell.kernel` is set but we patch the shell completer, so we can't
    # re-derive the branch from `shell.kernel` here.
    if _did_patch_shell_completer:
        unpatch_shell_completer(shell.Completer)
    else:
        unpatch_kernel_completer()
