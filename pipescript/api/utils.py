from __future__ import annotations

import functools
from types import FrameType
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from pipescript.analysis.placeholders import FreeVarTransformer
from pipescript.constants import pipeline_null

T = TypeVar("T")


# allow-listed print function that won't cause the no-prints check to fail
print_ = print


if TYPE_CHECKING:
    _dynamic_lookup: Callable[[str, int], Any]
else:

    @functools.cache
    def _dynamic_lookup(name: str, frame_id: int) -> Any:
        frame: FrameType | None = FreeVarTransformer.frame_cache.get(frame_id)
        while frame is not None and name not in frame.f_locals:
            frame = frame.f_back
        if frame is None:
            raise NameError("Undefined name '%s'" % name)
        return frame.f_locals[name]


def null(*_, **__) -> None:
    return None


def peek(obj: T, *args, **kwargs) -> T:
    print_(obj, *args, **kwargs)
    return obj


def collapse(results: tuple[T | None, ...]) -> T:
    filtered_results: list[T] = []
    for result in results:
        if result is None or result is pipeline_null:
            continue
        filtered_results.append(result)
    if len(filtered_results) != 1:
        raise ValueError(
            "Expected exactly one non-None result, got %d" % len(filtered_results)
        )
    else:
        return filtered_results[0]


stack: list[Any] = []


def push(obj: T) -> T:
    stack.append(obj)
    return obj


def pop(obj: T) -> tuple[T, Any]:
    return obj, stack.pop()


def lshift(obj: tuple[Any, ...]) -> tuple[Any, ...]:
    return obj[1:] + (obj[0],)


def rshift(obj: tuple[Any, ...]) -> tuple[Any, ...]:
    return (obj[-1],) + obj[:-1]


def unnest(obj: tuple[tuple[Any, ...], Any]) -> tuple[Any, ...]:
    return obj[0] + (obj[1],)


def replace(newval: T) -> Callable[..., T]:
    def __ignore(*_, **__) -> T:
        return newval

    return __ignore


__all__ = [
    "collapse",
    "lshift",
    "null",
    "peek",
    "pop",
    "push",
    "replace",
    "rshift",
    "unnest",
]
