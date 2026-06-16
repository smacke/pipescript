"""Tests for pipescript-aware runtime error enrichment.

Covers the runtime apply-site wrapper (precise footgun hints), the
``fork``/``parallel`` branch-index context, the block-source linecache mapping,
and the pure diagnostic helpers used by the display-time backstop.
"""

from __future__ import annotations

import linecache
import subprocess
import sys
import textwrap
import traceback
import types
from typing import Generator

import pyccolo as pyc
import pytest

from pipescript.patches.diagnostics import (
    add_note,
    annotate_pipescript_exception,
    diagnose,
)
from pipescript.tracers.brace_block_tracer import BraceBlockTracer
from pipescript.tracers.macro_tracer import MacroTracer
from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer
from pipescript.tracers.pipeline_tracer import PipelineTracer


@pytest.fixture(autouse=True)
def all_tracers() -> Generator[None, None, None]:
    MacroTracer.dynamic_macros.clear()
    MacroTracer.dynamic_method_macros.clear()
    with BraceBlockTracer:
        with PipelineTracer:
            with MacroTracer:
                with OptionalChainingTracer:
                    yield


def _notes(exc: BaseException) -> str:
    return "\n".join(getattr(exc, "_pyc_notes", []))


# --------------------------------------------------------------------------- #
# Runtime apply-site footgun hints                                            #
# --------------------------------------------------------------------------- #


def test_apply_where_compose_meant_gets_hint():
    # The second branch's `map[...] |> all` applies `all` to the `map[...]`
    # stage-function instead of composing.
    with pytest.raises(TypeError) as exc_info:
        pyc.eval(
            "[(1, 1)] |> fork[ map[$v[0] == $v[1]] .> any, "
            "map[$v[0] == $v[1]] |> all ]"
        )
    notes = _notes(exc_info.value)
    assert "function" in str(exc_info.value)
    assert ".>" in notes and "|>" in notes
    assert "compose" in notes and "apply" in notes


def test_fork_branch_index_in_note():
    with pytest.raises(TypeError) as exc_info:
        pyc.eval(
            "[(1, 1)] |> fork[ map[$v[0] == $v[1]] .> any, "
            "map[$v[0] == $v[1]] |> all ]"
        )
    notes = _notes(exc_info.value)
    # the *second* branch is the buggy one (1-based in the message)
    assert "fork branch #2 of 2" in notes


def test_full_motivating_snippet_diagnoses_and_fix_works():
    snippet = textwrap.dedent(
        """
        ["aaa", "bbb"] |> map{
            pairwise = zip($, $[1:]) |> list
            all_valid = pairwise |> fork[
                map[$v[0] == $v[1]] .> any,
                map["".join($) not in ("ab", "cd", "pq", "xy")] |> all
            ] |> all
            num_vowels = $ |> map[$ in "aeiou"] |> sum
            all_valid and num_vowels >= 3
        } |> sum
        """
    ).strip()
    with pytest.raises(TypeError) as exc_info:
        pyc.eval(snippet)
    notes = _notes(exc_info.value)
    assert "`|> all`" in notes  # names the offending stage
    assert "fork branch #2 of 2" in notes  # names the branch
    assert ".>" in notes

    # the documented fix (`|> all` -> `.> all` in the second branch) computes the
    # correct AoC-2015-day-5 nice-string count.
    assert pyc.eval(snippet.replace("] |> all\n", "] .> all\n", 1)) == 1


def test_unrelated_stage_error_is_not_mislabeled_as_footgun():
    # A genuine type error inside a stage (int + str) is *not* a leaked-function
    # footgun, so the apply-site wrapper must not attach a compose hint.
    with pytest.raises(TypeError) as exc_info:
        pyc.eval("[1] |> map[$ + 'x'] |> list")
    notes = _notes(exc_info.value)
    assert "compose" not in notes


# --------------------------------------------------------------------------- #
# Block source mapping                                                         #
# --------------------------------------------------------------------------- #


def test_block_traceback_points_at_original_source():
    with pytest.raises(AttributeError) as exc_info:
        pyc.eval("[1, 2] |> map{ y = $.no_such_attr; y }")
    # the block frame is renamed to a meaningful `map{...}` (not a synthetic
    # `__pyc_macro_block__`) and its source is registered for visibility
    block_frames = [
        (frame.f_code.co_filename, lineno, frame.f_code.co_name)
        for frame, lineno in traceback.walk_tb(exc_info.value.__traceback__)
        if frame.f_code.co_filename in MacroTracer._block_linecache_files
    ]
    assert block_frames, "expected a compiled-block frame in the traceback"
    fname, lineno, co_name = block_frames[-1]
    assert co_name == "map{...}"
    line = linecache.getline(fname, lineno)
    # the user's original `$` source, not the desugared `_0` sandbox form
    assert "$.no_such_attr" in line


# --------------------------------------------------------------------------- #
# Pure helpers (display-time backstop)                                         #
# --------------------------------------------------------------------------- #


def test_add_note_is_idempotent_and_ordered():
    exc = ValueError("boom")
    add_note(exc, "first")
    add_note(exc, "first")
    add_note(exc, "second")
    assert exc._pyc_notes == ["first", "second"]


def test_diagnose_callable_value_is_footgun_list_value_is_not():
    err = TypeError("'function' object is not iterable")
    assert diagnose(err, value=lambda: None) is not None
    assert diagnose(err, value=[1, 2, 3]) is None  # known non-callable, no hint


def test_diagnose_message_fallback_when_value_unknown():
    err = TypeError("'function' object is not iterable")
    note = diagnose(err, generic=True)  # backstop path: no value available
    assert note is not None and ".>" in note


def test_annotate_pipescript_exception_no_double_annotate():
    exc = TypeError("'function' object is not iterable")
    add_note(exc, "already here")
    annotate_pipescript_exception(TypeError, exc, None)
    assert exc._pyc_notes == ["already here"]  # untouched


def test_block_source_marks_traceback_visible():
    import pyccolo as pyc

    try:
        pyc.eval("[1, 2] |> map{ y = $.no_such_attr; y }")
    except AttributeError:
        pass
    # the compiled block's sandbox file is registered for traceback visibility
    assert any(pyc.is_traceback_visible(f) for f in MacroTracer._block_linecache_files)


def test_resugar_block_markers_rewrites_linecache():
    import linecache

    from pipescript.extension import resugar_block_markers

    fname = "<pyc-test-cell>"
    src = '["a"] | map[__pyc_block__(7)] | sum\n'
    linecache.cache[fname] = (len(src), None, [src], fname)
    try:
        fake = types.SimpleNamespace(
            tb_frame=types.SimpleNamespace(
                f_code=types.SimpleNamespace(co_filename=fname)
            ),
            tb_next=None,
        )
        resugar_block_markers(fake)
        assert linecache.cache[fname][2] == ['["a"] | map{...} | sum\n']
    finally:
        linecache.cache.pop(fname, None)


# --------------------------------------------------------------------------- #
# End-to-end IPython traceback pinpointing (subprocess, like test_reexecution) #
# --------------------------------------------------------------------------- #

_TB_PROBE = textwrap.dedent(
    """
    from IPython.testing.globalipapp import get_ipython
    import pipescript

    ip = get_ipython()
    pipescript.load_ipython_extension(ip)
    ip.run_cell("pass")  # warm up / load builtin dynamic macros
    CELL = (
        \'["aaa", "bbb"] |> map{\\n\'
        \'    pairwise = zip($, $[1:]) |> list\\n\'
        \'    all_valid = pairwise |> fork[\\n\'
        \'        map[$v[0] == $v[1]] .> any,\\n\'
        \'        map["".join($) not in ("ab", "cd") ] |> all\\n\'
        \'    ] |> all\\n\'
        \'    num_vowels = $ |> map[$ in "aeiou"] |> sum\\n\'
        \'    all_valid and num_vowels >= 3\\n\'
        \'} |> sum\'
    )
    ip.run_cell(CELL)
    """
)


def test_ipython_traceback_pinpoints_failing_stage():
    import os

    pytest.importorskip("IPython")
    # Run as a normal user would: dev mode disables traceback filtering and note
    # rendering, so explicitly clear it (pytest sets it process-wide).
    env = {k: v for k, v in os.environ.items() if k != "PYCCOLO_DEV_MODE"}
    proc = subprocess.run(
        [sys.executable, "-c", _TB_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    out = proc.stdout + proc.stderr
    # the traceback walks into the block and lands on the exact buggy stage
    assert 'map["".join($) not in ("ab", "cd") ] |> all' in out
    # ...kept visible despite the sandbox-frame filter, with the diagnosis notes
    assert "fork branch #2 of 2" in out
    assert ".>" in out and "compose" in out
    # block frame reads meaningfully and the desugared marker is re-sugared away
    assert "map{...}" in out
    assert "__pyc_block__" not in out
    assert "__pyc_macro_block__" not in out
