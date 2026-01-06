from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, TypeVar

from pipescript.constants import pipeline_null

T = TypeVar("T")
R = TypeVar("R")


# Like functoolz `do`
def do(func: Callable[[T, *tuple[T]], Any], obj: T, *extra: T) -> T | tuple[T, ...]:
    func(obj, *extra)
    return obj if len(extra) == 0 else (obj, *extra)


# Call multiple functions on an input and aggregate the results into a tuple
def fork(
    funcs: tuple[Callable[[T, *tuple[T]], Any]], obj: T, *extra: T
) -> tuple[Any, ...]:
    results = []
    for func in funcs:
        results.append(func(obj, *extra))
    return tuple(results)


# Call multiple functions on an input in parallel aggregate the results into a tuple bulk-synchronously
# Basically a parallel version of fork
def parallel(
    funcs: tuple[Callable[[T, *tuple[T]], Any]], obj: T, *extra: T
) -> tuple[Any, ...]:
    futures = []
    with ThreadPoolExecutor(max_workers=len(funcs)) as executor:
        for func in funcs:
            futures.append(executor.submit(func, obj, *extra))
    return tuple(fut.result() for fut in futures)


def when(func: Callable[[T, *tuple[T]], bool], obj: T, *extra: T) -> T | tuple[T, ...]:
    if func(obj, *extra):
        return obj if len(extra) == 0 else (obj, *extra)
    else:
        return pipeline_null  # type: ignore[return-value]


# just like `when` but inverted
def unless(
    func: Callable[[T, *tuple[T]], bool], obj: T, *extra: T
) -> T | tuple[T, ...]:
    return when(lambda o, *e: not func(o, *e), obj, *extra)


# `until` just an alias of `unless`
def until(func: Callable[[T, *tuple[T]], bool], obj: T, *extra: T) -> T | tuple[T, ...]:
    return unless(func, obj, *extra)


def repeat(func: Callable[[T, *tuple[T]], T], obj: T, *extra: T) -> T | tuple[T, ...]:
    __hide_pyccolo_frame__ = True
    ret = func(obj, *extra)
    if ret is pipeline_null:
        return obj if len(extra) == 0 else (obj, *extra)
    else:
        return (
            __hide_pyccolo_frame__ and repeat(func, ret)  # type: ignore[return-value]
            if len(extra) == 0
            else repeat(func, *ret)  # type: ignore[misc]
        )


def future(func: Callable[[T, *tuple[T]], R], obj: T, *extra: T) -> Future[R]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(func, obj, *extra)


memory: dict[str, Any] = {}


def write(key: str, obj: T) -> T:
    memory[key] = obj
    return obj


def read(key: str, obj: T) -> tuple[T, Any]:
    return obj, memory[key]


_ntimes_counters: dict[int, int] = {}


def ntimes(obj: T, callpoint_id: int) -> T:
    ctr = _ntimes_counters[callpoint_id]
    if ctr <= 0:
        return pipeline_null  # type: ignore[return-value]
    ctr -= 1
    _ntimes_counters[callpoint_id] = ctr
    return obj


__all__ = [
    "do",
    "fork",
    "future",
    "ntimes",
    "parallel",
    "read",
    "repeat",
    "unless",
    "until",
    "when",
    "write",
]
