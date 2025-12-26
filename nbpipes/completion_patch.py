from __future__ import annotations

from IPython.core.completer import Completer, Completion


def patch_completer(completer: Completer) -> None:
    clazz: type[Completer] = completer.__class__

    class PatchedCompleter(clazz):  # type: ignore[misc, valid-type]
        def completions(self, text: str, offset: int) -> list[Completion]:
            before_offset = text[:offset]
            placeholder_split = before_offset.rsplit("$.", 1)
            if len(placeholder_split) == 2:
                shift_amount = len(placeholder_split[0])
                completions = super().completions(
                    "_." + placeholder_split[1], offset - shift_amount
                )
                completions = list(completions)
                for completion in completions:
                    completion.start += shift_amount
                    completion.end += shift_amount
                return completions
            else:
                return super().completions(text, offset)

    completer.__class__ = PatchedCompleter


def unpatch_completer(completer: Completer) -> None:
    completer.__class__ = completer.__class__.mro()[1]
