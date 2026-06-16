from __future__ import annotations

import functools
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, TypeVar

from pipescript.constants import pipeline_null

T = TypeVar("T")
R = TypeVar("R")
C = TypeVar("C", bound=Callable[..., Any])


def _run_branch(
    func: Callable[..., Any],
    obj: Any,
    extra: tuple[Any, ...],
    label: str,
    idx: int,
    count: int,
) -> Any:
    """Call a ``fork``/``parallel`` branch, tagging any error with which branch
    raised. Pairs with the apply-vs-compose hint from the pipeline tracer, which
    says *what* went wrong; this says *which* branch."""
    __hide_pyccolo_frame__ = True  # noqa: F841
    try:
        return func(obj, *extra)
    except Exception as e:
        from pipescript.patches.diagnostics import add_note

        add_note(e, f"pipescript: ...in {label} branch #{idx + 1} of {count}")
        raise


# Like functoolz `do`
def do(func: Callable[[T, *tuple[T]], Any], obj: T, *extra: T) -> T | tuple[T, ...]:
    # Hidden so a co-tracer (e.g. ipyflow) walking the stack from the user
    # placeholder-lambda this calls skips this frame rather than mapping its
    # static_macros line number onto the executing cell. See `repeat` below.
    __hide_pyccolo_frame__ = True  # noqa: F841
    func(obj, *extra)
    return obj if len(extra) == 0 else (obj, *extra)


# Call multiple functions on an input and aggregate the results into a tuple
def fork(
    funcs: tuple[Callable[[T, *tuple[T]], Any]], obj: T, *extra: T
) -> tuple[Any, ...]:
    __hide_pyccolo_frame__ = True  # noqa: F841
    if isinstance(funcs[0], bool):
        # TODO: kinda hacky; is there a more robust way to signal the presence of an `otherwise` macro?
        return _fork_with_otherwise(funcs[1:], obj, *extra)
    results = []
    for i, func in enumerate(funcs):
        results.append(_run_branch(func, obj, extra, "fork", i, len(funcs)))
    return tuple(results)


# Call multiple functions on an input in parallel aggregate the results into a tuple bulk-synchronously
# Basically a parallel version of fork
def parallel(
    funcs: tuple[Callable[[T, *tuple[T]], Any]], obj: T, *extra: T
) -> tuple[Any, ...]:
    if isinstance(funcs[0], bool):
        return _parallel_with_otherwise(funcs[1:], obj, *extra)
    futures = []
    with ThreadPoolExecutor(max_workers=len(funcs)) as executor:
        for i, func in enumerate(funcs):
            futures.append(
                executor.submit(
                    _run_branch, func, obj, extra, "parallel", i, len(funcs)
                )
            )
    return tuple(fut.result() for fut in futures)


def when(func: Callable[[T, *tuple[T]], bool], obj: T, *extra: T) -> T | tuple[T, ...]:
    __hide_pyccolo_frame__ = True  # noqa: F841
    if func(obj, *extra):
        return obj if len(extra) == 0 else (obj, *extra)
    else:
        return pipeline_null  # type: ignore[return-value]


def otherwise(func: C) -> C:
    return func


def _fork_with_otherwise(
    funcs: tuple[Callable[[T, *tuple[T]], Any]], obj: T, *extra: T
) -> tuple[Any, ...]:
    __hide_pyccolo_frame__ = True  # noqa: F841
    results: list[Any] = []
    func: Callable[[T, *tuple[T]], Any]
    for i, func in enumerate(funcs[:-1]):
        results.append(_run_branch(func, obj, extra, "fork", i, len(funcs)))
    if all(res is pipeline_null for res in results):
        results.append(
            _run_branch(funcs[-1], obj, extra, "fork", len(funcs) - 1, len(funcs))
        )
    else:
        results.append(pipeline_null)
    return tuple(results)


def _parallel_with_otherwise(
    funcs: tuple[Callable[[T, *tuple[T]], Any]], obj: T, *extra: T
) -> tuple[Any, ...]:
    futures: list[Future[Any]] = []
    with ThreadPoolExecutor(max_workers=max(len(funcs) - 1, 32)) as executor:
        func: Callable[[T, *tuple[T]], Any]
        for i, func in enumerate(funcs[:-1]):
            futures.append(
                executor.submit(
                    _run_branch, func, obj, extra, "parallel", i, len(funcs)
                )
            )
    results = list(fut.result() for fut in futures)
    if all(res is pipeline_null for res in results):
        results.append(
            _run_branch(funcs[-1], obj, extra, "parallel", len(funcs) - 1, len(funcs))
        )
    else:
        results.append(pipeline_null)
    return tuple(results)


# just like `when` but inverted
def unless(
    func: Callable[[T, *tuple[T]], bool], obj: T, *extra: T
) -> T | tuple[T, ...]:
    __hide_pyccolo_frame__ = True  # noqa: F841

    # a nested ``def`` (not a lambda) so the negating wrapper frame can also carry
    # the hide marker -- otherwise a co-tracer stops at it and maps this file's
    # line onto the cell.
    def _negate(o: T, *e: T) -> bool:
        __hide_pyccolo_frame__ = True  # noqa: F841
        return not func(o, *e)

    return when(_negate, obj, *extra)


# `until` just an alias of `unless`
def until(func: Callable[[T, *tuple[T]], bool], obj: T, *extra: T) -> T | tuple[T, ...]:
    __hide_pyccolo_frame__ = True  # noqa: F841
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


def context(func, obj):
    __hide_pyccolo_frame__ = True  # noqa: F841
    with obj as o:
        return func(obj if o is None else o)


def expect(
    func: Callable[[T, *tuple[T]], bool], obj: T, *extra: T
) -> T | tuple[T, ...]:
    __hide_pyccolo_frame__ = True  # noqa: F841
    assert func(obj, *extra)
    return obj if len(extra) == 0 else (obj, *extra)


_once_cache: dict[int, Any] = {}


def once(callpoint_id: int) -> Any:
    return _once_cache[callpoint_id]


def memoize(func: C) -> C:
    return functools.cache(func)  # type: ignore[return-value]


__all__ = [
    "context",
    "do",
    "expect",
    "fork",
    "future",
    "memoize",
    "ntimes",
    "once",
    "otherwise",
    "parallel",
    "read",
    "repeat",
    "unless",
    "until",
    "when",
    "write",
]
