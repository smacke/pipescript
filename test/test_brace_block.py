from __future__ import annotations

from typing import Any, Generator

import pyccolo as pyc
import pytest

from pipescript.constants import pipeline_null
from pipescript.tracers.brace_block_tracer import BraceBlockTracer
from pipescript.tracers.macro_tracer import DynamicMacro, MacroTracer
from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer
from pipescript.tracers.pipeline_tracer import PipelineTracer


@pytest.fixture(autouse=True)
def all_tracers() -> Generator[None, None, None]:
    MacroTracer.dynamic_macros.clear()
    MacroTracer.dynamic_method_macros.clear()
    # BraceBlockTracer must be outermost so brace extraction happens before the
    # `$` -> `_` placeholder pass.
    with BraceBlockTracer:
        with PipelineTracer:
            with MacroTracer:
                with OptionalChainingTracer:
                    for (
                        name,
                        mdef,
                    ) in MacroTracer.builtin_dynamic_macro_definitions.items():
                        _refresh(pyc.exec(f"{name} = {mdef}"))
                    yield


def _refresh(env: dict[str, Any]) -> None:
    for k, v in env.items():
        if not isinstance(v, DynamicMacro):
            continue
        if v.is_method:
            MacroTracer.dynamic_method_macros[k] = v
        else:
            MacroTracer.dynamic_macros[k] = v
    MacroTracer.instance().reset()


def test_quick_lambda_statement_block_with_for_loop():
    # the motivating example: a multi-line lambda with a for-loop, $ = input
    assert (
        pyc.eval("5 |> f{\n  acc = 0\n  for i in range($):\n    acc += i\n  acc\n}")
        == 10
    )


def test_straight_line_block_binds_and_reuses_placeholder():
    # $ is captured once into a local, then reused -- something the single-$
    # expression form can't do without $$/named args
    assert pyc.eval("4 |> f{ a = $ + 1\n a * a }") == 25


def test_if_else_block():
    assert (
        pyc.eval(
            "7 |> f{\n  if $ % 2 == 0:\n    r = 'even'\n  else:\n    r = 'odd'\n  r\n}"
        )
        == "odd"
    )


def test_while_block():
    assert (
        pyc.eval(
            "1 |> f{\n  n = $\n  total = 0\n  while n <= 4:\n    total += n\n    n += 1\n  total\n}"
        )
        == 10
    )


def test_named_placeholder():
    assert pyc.eval("10 |> f{ half = $x // 2\n half + 1 }") == 6


def test_expression_brace_body_still_works():
    # an expression body is just a brace-for-bracket swap
    assert pyc.eval("5 |> f{ $ + 1 }") == 6


def test_do_tap_matches_bracket_form():
    # `do` is a tap (returns its input); the statement block matches the bracket form
    assert pyc.eval("9 |> do{ tmp = $ * 5\n tmp }") == pyc.eval("9 |> do[$ * 5]")


def test_when_matches_bracket_form():
    assert pyc.eval("6 |> when{ p = $ % 2\n p == 0 }") == pyc.eval(
        "6 |> when[$ % 2 == 0]"
    )
    assert pyc.eval("5 |> when{ p = $ % 2\n p == 0 }") == pyc.eval(
        "5 |> when[$ % 2 == 0]"
    )


def test_when_false_is_pipeline_null():
    assert pyc.eval("5 |> when{ p = $ % 2\n p == 0 }") in (pipeline_null, None)


def test_repeat_block():
    # `repeat` keeps applying until the block returns pipeline_null
    assert (
        pyc.eval(
            "0 |> repeat{\n  if $ < 3:\n    r = $ + 1\n  else:\n    r = pipeline_null\n  r\n}"
        )
        == 3
    )


def test_block_can_call_plain_functions():
    assert pyc.eval("3 |> f{ xs = list(range($))\n sum(xs) }") == 3


def test_method_macro_foreach_statement_block():
    # a dynamic *method* macro (foreach) with a statement block; $ is each item
    ns = pyc.exec(
        "out = []\n"
        "range(4).foreach{\n"
        "    v = $\n"
        "    out.append(v * v)\n"
        "}\n"
        "result = out"
    )
    assert ns["result"] == [0, 1, 4, 9]


def test_method_macro_block_mutates_enclosing_scope():
    # the block reads/mutates variables from the enclosing (non-global) scope
    moves = "><><"
    ns = pyc.exec(
        "houses = {(0, 0)}\n"
        "cur = [0, 0]\n"
        "offs = {'>': 1, '<': -1}\n"
        f"enumerate({moves!r}).foreach{{\n"
        "    i, d = $\n"
        "    idx = i % 2\n"
        "    cur[idx] += offs[d]\n"
        "    houses.add((idx, cur[idx]))\n"
        "}\n"
        "result = sorted(houses)"
    )
    # plain-Python reference
    houses = {(0, 0)}
    cur = [0, 0]
    offs = {">": 1, "<": -1}
    for i, d in enumerate(moves):
        idx = i % 2
        cur[idx] += offs[d]
        houses.add((idx, cur[idx]))
    assert ns["result"] == sorted(houses)


# --- nested pipescript syntax inside a statement block ---


def test_pipeline_inside_block():
    assert pyc.eval("5 |> f{ y = ($ |> f[$ + 1])\n y * 10 }") == 60


def test_bracket_macro_inside_block():
    assert pyc.eval("5 |> f{ y = ($ |> when[$ > 0])\n y }") == 5


def test_nested_expression_block_inside_block():
    assert pyc.eval("5 |> f{ y = ($ |> f{ $ + 1 })\n y * 10 }") == 60


def test_multiline_pipeline_chain_inside_block():
    assert (
        pyc.eval(
            "10 |> f{\n"
            "    half = ($ |> f[$ // 2])\n"
            "    bumped = (half |> f[$ + 1])\n"
            "    bumped * 100\n"
            "}"
        )
        == 600
    )


def test_nested_statement_block_inside_block():
    # a statement block nested inside another statement block
    assert pyc.eval("5 |> f{ y = ($ |> f{ a = $ + 1\n a })\n y * 10 }") == 60


def test_nested_statement_block_direct_call():
    assert pyc.eval("5 |> f{ z = f{ a = $ + 1\n a }(10)\n z * $ }") == 55
