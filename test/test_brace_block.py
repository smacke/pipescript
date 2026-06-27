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


def test_namespace_block_macro_harvests_assignments():
    from pipescript.tracers.macro_tracer import register_namespace_macro

    register_namespace_macro("record", lambda ns: ns)
    try:
        # top-level assignments become dict entries; `_tmp` is a local temporary
        # (excluded) but still usable within the block.
        out = pyc.eval("record{\n  a = 1\n  b = a + 2\n  _tmp = 99\n  c = _tmp + b\n}")
        assert out == {"a": 1, "b": 3, "c": 102}
    finally:
        MacroTracer.static_macros.pop("record", None)
        MacroTracer.namespace_block_macros.pop("record", None)
        __import__("builtins").__dict__.pop("record", None)


def test_namespace_block_macro_applies_builder_and_sees_free_vars():
    from pipescript.tracers.macro_tracer import register_namespace_macro

    register_namespace_macro("scaled", lambda ns: {k: v * 10 for k, v in ns.items()})
    try:
        env = {"base": 5}
        out = pyc.eval("scaled{\n  x = base\n  y = base + 1\n}", env, env)
        assert out == {"x": 50, "y": 60}  # builder applied; `base` resolved as free var
    finally:
        MacroTracer.static_macros.pop("scaled", None)
        MacroTracer.namespace_block_macros.pop("scaled", None)
        __import__("builtins").__dict__.pop("scaled", None)


def test_namespace_block_macro_supports_nested_pipescript():
    from pipescript.tracers.macro_tracer import register_namespace_macro

    register_namespace_macro("ns", lambda d: d)
    try:
        out = pyc.eval("ns{\n  evens = [1, 2, 3, 4] |> filter[$ % 2 == 0] |> list\n}")
        assert out == {"evens": [2, 4]}  # `|>` inside the block still dispatches
    finally:
        MacroTracer.static_macros.pop("ns", None)
        MacroTracer.namespace_block_macros.pop("ns", None)
        __import__("builtins").__dict__.pop("ns", None)


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


def test_method_macro_as_pipe_stage_with_placeholder_receiver():
    # `$.foreach[...]` as a pipe stage: the `$` receiver is the piped value and
    # must be bound by the stage lambda. Previously it evaluated as an unbound
    # name before the macro could fire ("name '_' is not defined").
    ns = pyc.exec(
        "out = []\n" "range(4) |> $.foreach[out.append($ ** 2)]\n" "result = out"
    )
    assert ns["result"] == [0, 1, 4, 9], ns["result"]


def test_method_macro_brace_block_as_pipe_stage_with_placeholder_receiver():
    # the brace-block form of the above, with tuple-unpacking in the body
    ns = pyc.exec(
        "positions = {}\n"
        "[('a', 1), ('b', 2), ('a', 3)] |> enumerate |> $.foreach{\n"
        "    i, kv = $\n"
        "    positions.setdefault(kv[0], []).append((i, kv[1]))\n"
        "}\n"
        "result = positions"
    )
    assert ns["result"] == {"a": [(0, 1), (2, 3)], "b": [(1, 2)]}, ns["result"]


def test_method_macro_pipe_stage_nested_in_block():
    # the user-reported case: a `... |> $.foreach{...}` pipeline nested inside an
    # enclosing `map{...}` block, whose own placeholder is distinct.
    ns = pyc.exec(
        "out = [['ab', 'cd'], ['ef']]\n"
        "result = out |> map{\n"
        "    counts = {}\n"
        "    $ |> enumerate |> $.foreach{\n"
        "        i, s = $\n"
        "        counts[i] = len(s)\n"
        "    }\n"
        "    counts\n"
        "} |> list"
    )
    assert ns["result"] == [{0: 2, 1: 2}, {0: 2}], ns["result"]


def test_block_marker_emission_is_idempotent():
    # A host like ipyflow runs the syntax augmenter several times per cell; the
    # same block must always get the same marker id, or the rewriter's
    # instrumentation (set up against one pass) fails to line up with the marker
    # the executed pass emits and the cell runs uninstrumented.
    tracer = BraceBlockTracer.instance()
    code = "xs |> map{ $ + 10 } |> list"
    first = tracer.preprocess(code, None)
    second = tracer.preprocess(code, None)
    assert first == second, (first, second)
    assert "__pyc_block__(" in first, first


def test_brace_untransform_roundtrip():
    # The brace rewrite rides pyccolo's custom-augmentation framework, so
    # ``untransform`` resugars ``macro[...]`` back to the ``macro{...}`` the user
    # wrote -- statement blocks recover their verbatim source, tuple templates
    # normalize spacing.
    t = OptionalChainingTracer.instance()
    expr_block = "xs |> map{ $ + 10 } |> list"
    assert t.untransform(t.parse(expr_block, instrument=False)) == expr_block
    stmt_block = "xs |> map{ acc = 0; acc + $ } |> list"
    assert t.untransform(t.parse(stmt_block, instrument=False)) == stmt_block
    chained = "ys |> map{ $ * 2 } |> filter{ $ > 3 } |> list"
    assert t.untransform(t.parse(chained, instrument=False)) == chained
    # fork/parallel tuple templates resugar (ast.unparse normalizes the spacing)
    assert t.untransform(t.parse("fork{ f1, f2 }", instrument=False)) == "fork{f1, f2}"


def test_brace_rewrite_threads_positions():
    # The old preprocess override dropped the positions argument; the custom
    # rewrite threads it, so a tracked location survives the brace rewrite.
    t = OptionalChainingTracer.instance()
    src = "xs |> map{ $ + 10 } |> list"
    out, positions = t.transform(src, positions=[(1, 0), (1, src.index("list"))])
    assert out.splitlines()[0][positions[0][1] : positions[0][1] + 2] == "xs"
    assert out.splitlines()[positions[1][0] - 1][positions[1][1] :].startswith("list")


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


# --- pure (analysis-only) transform + thread-local hardening ------------------

# A foreach statement block: lowered via `_emit`'s block path (the body isn't a
# top-level tuple), so it exercises the registration the bug report is about.
_BLOCK_SRC = "range(4).foreach{\n    v = $\n    acc.append(v)\n}"


def _analysis_tracers() -> list[pyc.BaseTracer]:
    # An explicit stack so the transform behaves identically regardless of which
    # thread calls it (mirrors how a consumer like DBLS resolves the live
    # singletons), instead of depending on the ambient `_TRACER_STACK`.
    return [
        BraceBlockTracer.instance(),
        PipelineTracer.instance(),
        MacroTracer.instance(),
        OptionalChainingTracer.instance(),
    ]


def test_pure_transform_of_block_leaves_state_unchanged():
    from pipescript.tracers.macro_tracer import BLOCK_MARKER_FUNC

    # Seed shared state as if a prior execution registered body #1.
    BraceBlockTracer._counter = 1
    BraceBlockTracer.block_sources.clear()
    BraceBlockTracer._id_by_source.clear()
    BraceBlockTracer.block_sources[1] = "<live body the kernel will read>"
    BraceBlockTracer._id_by_source["<live body the kernel will read>"] = 1
    before_counter = BraceBlockTracer._counter
    before_sources = dict(BraceBlockTracer.block_sources)
    before_ids = dict(BraceBlockTracer._id_by_source)

    out = pyc.transform(_BLOCK_SRC, tracers=_analysis_tracers(), pure=True)

    # lowered to a valid, lintable marker referencing the no-op block id 0...
    assert isinstance(out, str)
    assert f"{BLOCK_MARKER_FUNC}(0)" in out
    # ...and the analysis left execution-relevant state byte-for-byte unchanged
    assert BraceBlockTracer._counter == before_counter
    assert dict(BraceBlockTracer.block_sources) == before_sources
    assert dict(BraceBlockTracer._id_by_source) == before_ids
    # the pure flag does not leak past the call
    assert pyc.is_pure_transform() is False


def test_default_transform_of_block_still_registers_body():
    # Backward compatibility: a normal (non-pure) transform registers the body.
    from pipescript.tracers.macro_tracer import BLOCK_MARKER_FUNC

    BraceBlockTracer._counter = 0
    BraceBlockTracer.block_sources.clear()
    BraceBlockTracer._id_by_source.clear()

    out = pyc.transform(_BLOCK_SRC, tracers=_analysis_tracers())  # pure defaults False

    assert f"{BLOCK_MARKER_FUNC}(1)" in out
    assert BraceBlockTracer._counter == 1
    assert set(BraceBlockTracer.block_sources) == {1}


def test_block_state_is_thread_local_under_concurrent_transform():
    # Regression for the reported race: a lint-style transform on a background
    # thread must not perturb the body the execution thread registered. Per-thread
    # storage makes this hold even when the consumer forgets `pure=True`.
    import threading

    BraceBlockTracer._counter = 0
    BraceBlockTracer.block_sources.clear()
    BraceBlockTracer._id_by_source.clear()

    ns = pyc.exec("acc = []\n" + _BLOCK_SRC + "\nresult = acc")
    assert ns["result"] == [0, 1, 2, 3]
    main_sources = dict(BraceBlockTracer.block_sources)
    assert main_sources  # the execution thread registered a body

    errors: list[BaseException] = []

    def _lint() -> None:
        try:
            # a *different* document, transformed WITHOUT pure mode
            other = "range(9).foreach{\n    w = $\n    sink.append(w * 2)\n}"
            pyc.transform(other, tracers=_analysis_tracers())
        except BaseException as exc:  # pragma: no cover - surfaced via assert
            errors.append(exc)

    t = threading.Thread(target=_lint)
    t.start()
    t.join()
    assert not errors, errors

    # the execution thread's registry is exactly as it left it...
    assert dict(BraceBlockTracer.block_sources) == main_sources
    # ...and re-executing the original block still resolves the right body
    ns2 = pyc.exec("acc = []\n" + _BLOCK_SRC + "\nresult = acc")
    assert ns2["result"] == [0, 1, 2, 3]
