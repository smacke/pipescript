from __future__ import annotations

import sys
from types import FrameType
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import pyccolo as pyc

from pipescript.constants import pipeline_null

T = TypeVar("T")


# allow-listed print function that won't cause the no-prints check to fail
print_ = print


if TYPE_CHECKING:
    _dynamic_lookup: Callable[[str], Any]
    _dynamic_store: Callable[[str, Any], Any]
else:

    def _resolve_enclosing_frame(name: str) -> FrameType | None:
        """Find the nearest enclosing *user* frame that binds ``name``.

        Called only from pipescript-synthesized code (a placeholder/branch lambda
        or a compiled block body) where ``name`` is a free variable of an
        enclosing function scope. The walk starts one frame above the immediate
        caller -- the synthesized frame itself is never the answer (the block's
        own seeded locals live there), the enclosing binding is always further
        out -- and skips the machinery frames pipescript/pyccolo mark with
        ``__hide_pyccolo_frame__`` so an intermediate frame that happens to bind a
        same-named local (e.g. the apply site's ``func``/``node`` params) can't
        shadow the user's binding. pyccolo's bare binop/unaryop lambdas cannot
        carry the marker, so their parameters are separately named to avoid
        clashing with user locals.

        Resolving against the live stack (rather than a frame captured at compile
        time) keeps reads current across repeated block/branch invocations, needs
        no frame retention, and -- crucially for write-back -- always targets a
        live frame."""
        # Depth 2 is the synthesized caller (0 = here, 1 = _dynamic_lookup/store);
        # start one frame above it so the block's own seeded locals never match.
        synthesized = sys._getframe(2)
        frame: FrameType | None = synthesized.f_back
        while frame is not None:
            if (
                "__hide_pyccolo_frame__" not in frame.f_locals
                and name in frame.f_locals
            ):
                return frame
            frame = frame.f_back
        return None

    def _dynamic_lookup(name: str) -> Any:
        frame = _resolve_enclosing_frame(name)
        if frame is None:
            raise NameError("Undefined name '%s'" % name)
        return frame.f_locals[name]

    def _dynamic_store(name: str, value: Any) -> Any:
        """Assign ``name = value`` back into the nearest enclosing frame that
        owns ``name``, so a macro block body can rebind an enclosing function
        local (e.g. ``total = total + $``). Returns ``value`` so the store can be
        used as an expression."""
        frame = _resolve_enclosing_frame(name)
        if frame is None:
            raise NameError("Undefined name '%s'" % name)
        pyc.set_frame_local(frame, name, value)
        return value


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
