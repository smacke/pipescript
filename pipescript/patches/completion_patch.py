from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import pyccolo as pyc

from pipescript.analysis.pipeline_lowering import lower_pipelines

if TYPE_CHECKING:
    from ipykernel.ipkernel import IPythonKernel
    from IPython.core.completer import Completer, Completion
    from IPython.core.interactiveshell import InteractiveShell


do_complete_patch_cls: type[IPythonKernel] = None  # type: ignore[assignment]
get_completion_context_patch_cls: type[IPythonKernel] = None  # type: ignore[assignment]
orig_do_complete = None
orig_get_completion_context = None
_did_patch_shell_completer = False


def _completion_token(text: str) -> tuple[str, str]:
    """The identifier being completed at the end of ``text``, plus the character
    that precedes it (``.`` for an attribute, ``(`` inside a call, ...)."""
    idx = len(text)
    while idx > 0 and (text[idx - 1].isalnum() or text[idx - 1] == "_"):
        idx -= 1
    return text[idx - 1] if idx > 0 else "", text[idx:]


def _tail_word_preserved(before_offset: str, transformed: str) -> bool:
    """Whether ``transformed`` ends in the same completion token as the original.

    Every caller completes at ``len(transformed)`` and shifts the returned offsets
    by ``cursor_pos - len(transformed)``; that arithmetic is only sound if the
    rewrite left the token under the cursor alone. Today's ``$`` -> ``_`` rewrite
    happens to be 1:1, so this held by accident. Making it a checked precondition
    lets us hand the completer arbitrarily rewritten source.
    """
    return _completion_token(before_offset) == _completion_token(transformed)


def _completion_sources(before_offset: str, tracers: list[pyc.BaseTracer]) -> list[str]:
    """Sources to hand the completer, in preference order.

    The statically lowered pipeline comes first: jedi can type it without the
    pipeline ever having run. The plain lexical transform is always last, so a
    completion that works today (via whatever the runtime ``_`` holds) is never
    lost. ``pure=True`` because this is analysis, not execution -- without it every
    keystroke in a cell holding a ``macro{...}`` block registers a block body.
    """
    legacy = pyc.transform(before_offset, tracers=tracers, pure=True)
    lowered = lower_pipelines(before_offset)
    if lowered is None:
        return [legacy]
    static = pyc.transform(lowered, tracers=tracers, pure=True)
    if static == legacy or not _tail_word_preserved(before_offset, static):
        return [legacy]
    return [static, legacy]


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
        sources = _completion_sources(before_offset, tracers)
        if len(sources) == 1 and sources[0] == before_offset:
            return orig_do_complete(self, code, cursor_pos)
        for i, source in enumerate(sources):
            completions = orig_do_complete(self, source, len(source))
            if completions.get("matches") or i == len(sources) - 1:
                break
        shift_amount = cursor_pos - len(source)
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
            orig_get_completion_context(self, *args, **kwargs),
            tracers=tracers,
            pure=True,
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
            sources = _completion_sources(before_offset, tracers)
            if len(sources) == 1 and sources[0] == before_offset:
                return super().completions(text, offset)
            for i, source in enumerate(sources):
                completions = list(super().completions(source, len(source)))
                if completions or i == len(sources) - 1:
                    break
            shift_amount = offset - len(source)
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
