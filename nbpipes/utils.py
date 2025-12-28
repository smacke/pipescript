from __future__ import annotations

from contextlib import contextmanager
from typing import TypeVar

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
