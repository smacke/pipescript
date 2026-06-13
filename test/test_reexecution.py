# -*- coding: utf-8 -*-
"""End-to-end re-execution test through a real IPython shell.

Re-running the same brace-block cell exercises the extension's full
re-execution flow (sandbox-filename allocation across tracer subclasses, the
ast-bookkeeping lifecycle) -- which a bare ``pyc.exec`` does not reproduce. We
run it in a *fresh subprocess* rather than the in-process global IPython shell,
because that singleton leaks tracer/extension state across the test session and
makes the outcome nondeterministic.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# The block's nested pipeline sits in the *final* branch so it only runs for the
# last element -- after the earlier elements' block calls have churned the
# process-wide ast bookkeeping. With colliding sandbox filenames across tracer
# subclasses, a later compile evicts the live block's bookkeeping and the inner
# ``$ |> $ + 1`` stage stops being recognized on the *second* execution
# -> "'function' object is not subscriptable".
_PROBE = textwrap.dedent(
    """
    import sys
    from IPython.testing.globalipapp import get_ipython
    import pipescript

    ip = get_ipython()
    pipescript.load_ipython_extension(ip)
    ip.run_cell("pass")  # load builtin dynamic macros (foreach, ...)
    CELL = (
        "out = []\\n"
        "[0, 1, 2].foreach{\\n"
        "    if $ == 0:\\n"
        "        out.append('zero')\\n"
        "    elif $ == 1:\\n"
        "        out.append('one')\\n"
        "    else:\\n"
        "        out.append($ |> $ + 1)\\n"
        "}\\n"
        "result = out"
    )
    for i in range(3):
        r = ip.run_cell(CELL)
        if r.error_in_exec is not None:
            print("FAIL run %d: %r" % (i, r.error_in_exec))
            sys.exit(1)
        if ip.user_ns.get("result") != ["zero", "one", 3]:
            print("FAIL run %d: result=%r" % (i, ip.user_ns.get("result")))
            sys.exit(1)
    print("OK")
    """
)


def test_block_with_nested_pipeline_survives_reexecution():
    pytest.importorskip("IPython")
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0 and proc.stdout.strip().endswith("OK"), (
        proc.stdout + proc.stderr
    )
