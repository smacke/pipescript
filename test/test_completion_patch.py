from __future__ import annotations

from typing import Any, cast

import pyccolo as pyc

from pipescript.patches import completion_patch as cp
from pipescript.patches.completion_patch import patch_completer, unpatch_completer
from pipescript.tracers.brace_block_tracer import BraceBlockTracer
from pipescript.tracers.macro_tracer import MacroTracer
from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer
from pipescript.tracers.pipeline_tracer import PipelineTracer


def _tracers() -> list[pyc.BaseTracer]:
    return [
        cast(pyc.BaseTracer, cls).instance()
        for cls in [
            BraceBlockTracer,
            PipelineTracer,
            MacroTracer,
            OptionalChainingTracer,
        ]
    ]


class FakeCompleter:
    def completions(self, text: str, offset: int) -> list[Any]:
        return [(text, offset)]


class ShellNoKernel:
    def __init__(self) -> None:
        self.Completer = FakeCompleter()


class KernelWithDoComplete:
    def do_complete(self, code: str, cursor_pos: int) -> dict[str, Any]:
        return {
            "matches": [],
            "cursor_start": 0,
            "cursor_end": cursor_pos,
            "metadata": {},
            "status": "ok",
        }


class PyodideLikeKernel:
    """Mimics jupyterlite's ``PyodideKernel``: no ``do_complete`` anywhere in
    its MRO, so ``patch_kernel_completer`` must decline to patch it."""


class ShellWithKernel:
    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel
        self.Completer = FakeCompleter()


def test_no_kernel_patches_shell_completer() -> None:
    shell = ShellNoKernel()
    orig_cls = shell.Completer.__class__
    patch_completer(shell, _tracers())
    try:
        assert cp._did_patch_shell_completer is True
        assert shell.Completer.__class__ is not orig_cls
        # the wrapped completer still delegates through to the base impl
        assert shell.Completer.completions("x", 1) == [("x", 1)]
    finally:
        unpatch_completer(shell)
    assert shell.Completer.__class__ is orig_cls


def test_kernel_with_do_complete_is_wrapped() -> None:
    shell = ShellWithKernel(KernelWithDoComplete())
    orig_do_complete = KernelWithDoComplete.do_complete
    patch_completer(shell, _tracers())
    try:
        assert cp._did_patch_shell_completer is False
        assert cp.do_complete_patch_cls is KernelWithDoComplete
        assert KernelWithDoComplete.do_complete is not orig_do_complete
        # shell completer left untouched on the kernel path
        assert shell.Completer.__class__ is FakeCompleter
    finally:
        unpatch_completer(shell)
    assert KernelWithDoComplete.do_complete is orig_do_complete
    assert cp.do_complete_patch_cls is None


def test_pyodide_like_kernel_falls_back_to_shell_completer() -> None:
    shell = ShellWithKernel(PyodideLikeKernel())
    orig_cls = shell.Completer.__class__
    # must not raise AttributeError on object.do_complete
    patch_completer(shell, _tracers())
    try:
        assert cp._did_patch_shell_completer is True
        assert shell.Completer.__class__ is not orig_cls
        assert shell.Completer.completions("x", 1) == [("x", 1)]
    finally:
        unpatch_completer(shell)
    assert shell.Completer.__class__ is orig_cls
