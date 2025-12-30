from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, TypeVar

T = TypeVar("T")


# allow-listed print function that won't cause the no-prints check to fail
print_ = print


def get_user_ns() -> dict[str, Any] | None:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None:
            return shell.user_ns
    except ImportError:
        pass
    return None


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
def do(func: Callable[[T], Any], obj: T) -> T:
    func(obj)
    return obj


# Call multiple functions on an input and aggregate the results into a tuple
def fork(funcs: tuple[Callable[[T], Any]], obj: T) -> tuple[Any]:
    results = []
    for func in funcs:
        results.append(func(obj))
    return tuple(results)


pipeline_null = object()


def when(func: Callable[[T], bool], obj: T) -> T:
    if func(obj):
        return obj
    else:
        return pipeline_null  # type: ignore[return-value]


def collapse(results: tuple[T | None, ...]) -> T:
    filtered_results: list[T] = []
    for result in results:
        if result is not None:
            filtered_results.append(result)
    if len(filtered_results) != 1:
        raise ValueError(
            "Expected exactly one non-None result, got %d" % len(filtered_results)
        )
    else:
        return filtered_results[0]
