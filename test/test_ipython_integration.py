"""End-to-end test that the ``%load_ext pipescript`` path actually instruments
user cells -- exercising the real IPython extension wiring (input transformers,
ast transformers, and per-cell filename binding) rather than driving the tracers
directly.

This guards against integration regressions like the IPython 9 cell-name
off-by-one, where the extension loaded but ``|>`` silently degraded to a plain
bitwise-or because the rewriter treated user cells as untraced.
"""
from __future__ import annotations

import pytest

pytest.importorskip("IPython")


def _fresh_shell():
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    result = shell.run_cell("%load_ext pipescript", store_history=True)
    # An extension that fails to load leaves ``|>`` as plain Python, so every
    # assertion below would instead die of an unrelated-looking SyntaxError.
    assert result.error_in_exec is None, result.error_in_exec
    return shell


def _run(shell, code: str):
    result = shell.run_cell(code, store_history=True)
    # A cell whose syntax augmenters never ran fails *before* exec, so checking
    # only ``error_in_exec`` would let a SyntaxError through as a ``None`` value.
    assert result.error_before_exec is None, result.error_before_exec
    assert result.error_in_exec is None, result.error_in_exec
    return result.result


def test_pipe_is_instrumented_via_load_ext():
    shell = _fresh_shell()
    try:
        assert _run(shell, "(3, 4, 1, 5, 6) |> sorted |> tuple") == (1, 3, 4, 5, 6)
        # a couple more cells to ensure it keeps working across the notebook
        assert _run(shell, "range(1, 5) |> reduce[$ * $]") == 24
        assert _run(shell, "5 |> fork{ $ + 1, $ * 2 }") == (6, 10)
    finally:
        from IPython.core.interactiveshell import InteractiveShell

        InteractiveShell.clear_instance()


def test_pipe_into_placeholder_expression():
    """``1 |> $ + 1`` -- piping into an expression rather than a callable.

    On IPython < 9 the pipe used to degrade to a bitwise-or against IPython's
    ``_`` last-output variable (``1 | _ + 1``), because ``ExecutionInfo`` there
    carries no ``transformed_cell`` and pyccolo dropped the source phase's
    rewriter along with every position its augmenters had registered.
    """
    shell = _fresh_shell()
    try:
        assert _run(shell, "1 |> $ + 1") == 2
        assert _run(shell, "10 |> $ * 2 |> $ - 5") == 15
    finally:
        from IPython.core.interactiveshell import InteractiveShell

        InteractiveShell.clear_instance()
