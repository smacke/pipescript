from __future__ import annotations

from typing import Any


# for unittest.mock patching
def _get_user_ns_impl() -> dict[str, Any] | None:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None:
            return shell.user_ns
    except ImportError:
        pass
    return None


def get_user_ns() -> dict[str, Any] | None:
    return _get_user_ns_impl()
