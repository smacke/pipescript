from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

from nbpipes.constants import pipeline_null

T = TypeVar("T")
R = TypeVar("R")


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


def when(func: Callable[[T, *tuple[T]], bool], obj: T, *extra: T) -> T | tuple[T, ...]:
    if func(obj, *extra):
        return obj if len(extra) == 0 else (obj, *extra)
    else:
        return pipeline_null  # type: ignore[return-value]


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


def future(func: Callable[[T, *tuple[T]], R], obj: T, *extra: T) -> Future[R]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(func, obj, *extra)


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
