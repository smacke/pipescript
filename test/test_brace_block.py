from __future__ import annotations

import ast
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


def test_bare_pipeline_stage_inside_block():
    # the pipe stage `$ + 1` is a *bare* expression (not wrapped in f[...]): its
    # `$` is the pipe's argument and must be left for PipelineTracer, while the
    # pipe input `$` collapses to the block input. Previously both `$` collapsed
    # to `_0`, yielding `2 |> 3` -> `3(2)` -> "int object is not callable".
    assert pyc.eval("2 |> f{ $ |> $ + 1 }") == 3


def test_chained_bare_pipeline_inside_block():
    assert pyc.eval("5 |> f{ $ |> $ * 2 |> $ + 1 }") == 11


def test_bare_pipeline_then_block_placeholder_reuse():
    # `$ * 10` after the pipeline statement is the block input again (the pipe
    # stage does not leak past the newline).
    assert pyc.eval("3 |> f{ x = ($ |> $ + 1)\n x + $ * 10 }") == 34


def test_bare_pipeline_stage_in_call_arg():
    # a pipe stage inside a call arg; the sibling arg's `$` is the block input
    assert pyc.eval("2 |> f{ max($ |> $ + 1, $) }") == 3


def test_foreach_block_with_bare_pipeline():
    # the user-reported case: a bare pipeline in a foreach statement block
    ns = pyc.exec(
        "seen = []\n"
        "[0, 1, 2].foreach{\n"
        "    if $ == 0:\n"
        "        seen.append('zero')\n"
        "    else:\n"
        "        seen.append($ |> $ + 1)\n"
        "}\n"
        "result = seen"
    )
    assert ns["result"] == ["zero", 2, 3], ns["result"]


def test_nested_statement_block_inside_block():
    # a statement block nested inside another statement block
    assert pyc.eval("5 |> f{ y = ($ |> f{ a = $ + 1\n a })\n y * 10 }") == 60


def test_nested_statement_block_direct_call():
    assert pyc.eval("5 |> f{ z = f{ a = $ + 1\n a }(10)\n z * $ }") == 55


def test_fork_tuple_body_via_braces():
    # a tuple body is the fork/parallel multi-function template -> expression path
    assert pyc.eval("5 |> fork{ $ + 1, $ * 2 }") == (6, 10)


def test_nested_method_macro_scopes_inner_placeholder():
    # the inner foreach's $ binds to the inner element, not the outer block's
    ns = pyc.exec(
        "seen = []\n"
        "outer = [(0, [10, 20]), (1, [30])]\n"
        "outer.foreach{\n"
        "    i, items = $\n"
        "    items.foreach{ seen.append(($, i)) }\n"
        "}\n"
        "result = seen"
    )
    assert ns["result"] == [(10, 0), (20, 0), (30, 1)], ns["result"]


def test_block_marker_emission_is_idempotent():
    # A host like ipyflow runs the syntax augmenter several times per cell; the
    # same block must always get the same marker id, or the rewriter's
    # instrumentation (set up against one pass) fails to line up with the marker
    # the executed pass emits and the cell runs uninstrumented.
    tracer = BraceBlockTracer.instance()
    code = "xs |> map{ $ + 10 } |> list"
    first = tracer._augment(code)
    second = tracer._augment(code)
    assert first == second, (first, second)
    assert "__pyc_block__(" in first, first


def test_block_marker_is_a_resolvable_sentinel_call():
    # The marker is a call to a *defined* sentinel builtin (not a bare undefined
    # name): a host that eagerly evaluates the slice before the macro substitutes
    # it must not hit a NameError. The sentinel resolves to None on its own.
    import builtins

    from pipescript.tracers.macro_tracer import BLOCK_MARKER_FUNC, block_marker_id

    assert getattr(builtins, BLOCK_MARKER_FUNC)(123) is None
    node = ast.parse(f"m[{BLOCK_MARKER_FUNC}(7)]", mode="eval").body
    assert block_marker_id(node) == 7
    assert block_marker_id(ast.parse("m[7]", mode="eval").body) is None


def test_method_macro_survives_bookkeeping_reset():
    # A method macro is defined once and may be expanded many cells later. A host
    # (e.g. ipyflow) wipes pyccolo's process-wide augmentation bookkeeping between
    # cells via `reset_bookkeeping`, which would otherwise strip the template's
    # `$$` / `|>` / nested-macro marks and make expansion silently no-op. The
    # template latches its marks at definition and re-establishes them per
    # expansion, so it keeps working after a reset.
    ns = pyc.exec("myeach = method[$$ |> map[do[$$]] |> list]")
    myeach = ns["myeach"]
    MacroTracer.dynamic_method_macros["myeach"] = myeach
    # simulate the host clearing all augmentation marks between cells
    for ids in pyc.BaseTracer.augmented_node_ids_by_spec.values():
        ids.clear()
    out = pyc.exec("seen = []\n[1, 2, 3].myeach[do[seen.append($)]]\nresult = seen")
    assert out["result"] == [1, 2, 3], out["result"]
