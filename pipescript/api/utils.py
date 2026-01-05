from __future__ import annotations

from contextlib import contextmanager
from typing import Any, TypeVar

from pipescript.constants import pipeline_null

T = TypeVar("T")


# allow-listed print function that won't cause the no-prints check to fail
print_ = print


def allow_pipelines_in_loops_and_calls(func=None):
    if func is None or not callable(func):

        @contextmanager
        def nothing():
            yield

        return nothing()
    else:
        return func


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


__all__ = [
    "allow_pipelines_in_loops_and_calls",
    "collapse",
    "lshift",
    "null",
    "peek",
    "pop",
    "push",
    "rshift",
    "unnest",
]
