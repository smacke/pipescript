from __future__ import annotations

from typing import TYPE_CHECKING

import pyccolo as pyc

if TYPE_CHECKING:
    from IPython.core.completer import Completer, Completion


def patch_completer(completer: Completer, tracers: list[pyc.BaseTracer]) -> None:
    clazz: type[Completer] = completer.__class__

    class PatchedCompleter(clazz):  # type: ignore[misc, valid-type]
        def completions(self, text: str, offset: int) -> list[Completion]:
            before_offset = text[:offset]
            transformed = pyc.transform(before_offset, tracers=tracers)
            if transformed == before_offset:
                return super().completions(text, offset)
            shift_amount = offset - len(transformed)
            completions = list(super().completions(transformed, len(transformed)))
            for completion in completions:
                completion.start += shift_amount
                completion.end += shift_amount
            return completions

    completer.__class__ = PatchedCompleter


def unpatch_completer(completer: Completer) -> None:
    completer.__class__ = completer.__class__.mro()[1]
