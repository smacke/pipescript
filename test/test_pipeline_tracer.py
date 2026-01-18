from __future__ import annotations

import textwrap
from typing import Generator
from unittest.mock import patch

import pyccolo as pyc
import pytest

import pipescript.utils
from pipescript.analysis.placeholders import SingletonArgCounterMixin
from pipescript.tracers.macro_tracer import MacroTracer
from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer
from pipescript.tracers.pipeline_tracer import PipelineTracer


@pytest.fixture(autouse=True)
def all_tracers() -> Generator[None, None, None]:
    with PipelineTracer:
        with MacroTracer:
            with OptionalChainingTracer:
                yield


def test_simple_pipeline():
    assert pyc.eval("(1, 2, 3) |> list") == [1, 2, 3]


def test_value_first_partial_apply_then_apply():
    assert pyc.eval("5 $> isinstance <| int") is True


def test_fake_infix():
    assert pyc.eval("5 $>isinstance<| int") is True


def test_value_first_partial_tuple_apply_then_apply():
    assert pyc.eval("(1, 2) *$> (lambda a, b, c: a + b + c) <| 3") == 6


def test_value_first_partial_tuple_apply_then_apply_quick_lambda():
    assert pyc.eval("(1, 2) *$> f[$ + $ + $] <| 3") == 6


def test_function_first_partial_apply_then_apply():
    assert pyc.eval("isinstance <$ 5 <| int") is True


def test_function_first_partial_tuple_apply_then_apply():
    assert pyc.eval("(lambda a, b, c: a + b + c) <$* (1, 2) <| 3") == 6


def test_function_first_partial_tuple_apply_then_apply_quick_lambda():
    assert pyc.eval("f[$ + $ + $] <$* (1, 2) <| 3") == 6


def test_pipe_into_value_first_partial_apply():
    assert pyc.eval("int |> (5 $> isinstance)") is True


def test_pipe_into_function_first_partial_apply():
    assert pyc.eval("int |> (isinstance <$ 5)") is True


def test_simple_pipeline_with_quick_lambda_map():
    assert pyc.eval("(1, 2, 3) |> f[map(f[$ + 1], $)] |> list") == [2, 3, 4]


def test_pipeline_assignment():
    assert pyc.eval(
        "(1, 2, 3) |> list |>> result |> f[map(f[$ + 1], $)] |> list |> f[result + $]"
    ) == [1, 2, 3, 2, 3, 4]


def test_pipeline_methods():
    assert pyc.eval("(1, 2, 3) |> list |> $.index(2)") == 1


def test_pipeline_methods_nonstandard_whitespace():
    assert pyc.eval("(1, 2, 3)   |>     list  |>      $.index(2)") == 1


def test_left_tuple_apply():
    assert pyc.eval("(5, int) *|> isinstance") is True


def test_right_tuple_apply():
    assert pyc.eval("isinstance <|* (5, int)") is True


def test_compose_op():
    assert pyc.eval("((lambda x: x * 5) <. (lambda x: x + 2))(10)") == 60


def test_tuple_compose_op():
    assert pyc.eval("((lambda x, y: x * 5 + y) <.* (lambda x: (x, x + 2)))(10)") == 62


def test_compose_op_no_space():
    assert pyc.eval("((lambda x: x * 5)<. (lambda x: x + 2))(10)") == 60


def test_compose_op_extra_space():
    assert pyc.eval("((lambda x: x * 5)  <. (lambda x: x + 2))(10)") == 60


def test_compose_op_with_parenthesized_quick_lambdas():
    assert pyc.eval("((f[$ * 5]) <. (f[$ + 2]))(10)") == 60


def test_compose_op_with_quick_lambdas():
    assert pyc.eval("(f[$ * 5] <. f[$ + 2])(10)") == 60


def test_pipeline_inside_quick_lambda():
    assert pyc.eval("2 |> f[$ |> $ + 2]") == 4
    assert pyc.eval("2 |> f[$ |> f[$ + 2]]") == 4


def test_pipeline_dot_op_with_optional_chain():
    assert (
        pyc.eval("(3, 1, 2) |> (list <. reversed <. sorted) |> $.index(2).?foo") is None
    )


def test_function_placeholder():
    # TODO: the commented out ones don't work due to an issue in how NamedExpr values don't get
    #   bound to lambda closures, which is a weakness in pyccolo BEFORE_EXPR_EVENTS. Technically
    #   BEFORE_EXPR_EVENTS should all be using the default value binding trick.
    # assert pyc.eval("(add := (lambda x, y: x + y)) and (add1 := add($, 1)) and add1(42)") == 43
    # assert pyc.eval("(add := (lambda x, y: x + y)) and add(42, 1)") == 43
    pyc.exec("(add := (lambda x, y: x + y)); assert add(42, 1) == 43")
    pyc.exec("(add := (lambda x, y: x + y)); assert (lambda y: add(42, y))(1)")
    pyc.exec("(add := (lambda x, y: x + y)); assert (lambda y: add(42, y)) <| 1 == 43")
    pyc.exec("(add := (lambda x, y: x + y)); assert add(42, $) <| 1 == 43")
    pyc.exec("(add := f[$ + $]); assert (add($, 1) <| 1) == 2")
    pyc.exec("(add := f[$ + $]); assert 1 |> add($, 1) == 2")
    pyc.exec("add = f[$ + $]; add1 = add($, 1); assert add1(42) == 43")
    pyc.exec("add = f[$ + $]; assert add($, 42) <| 1 == 43")
    assert pyc.eval("(f[$ + $] |>> add) and add($, 1) <| 1") == 2
    assert pyc.eval("(f[$ + $] |>> add) and 1 |> add($, 1)") == 2


def test_tuple_unpack_with_placeholders():
    assert pyc.eval("($, $) *|> $ + $ <|* (1, 2)") == 3
    assert pyc.eval("($, $) *|> $ + $ <|* (1, 2) |> $.real") == 3
    assert pyc.eval("($, $) *|> $ + $ <|* (1, 2) |> $.imag") == 0
    assert pyc.eval("($, $) *|> $ + $ <|* (1, 2) |> $ + 1") == 4
    assert pyc.eval("(1, 2) *|> ($, $) *|> $ + $") == 3


def test_placeholder_with_kwarg():
    pyc.exec(
        textwrap.dedent(
            """
            def add(x, y):
                return x + y
            1 |> add($, y=42) |> expect[$ == 43]
            """.strip(
                "\n"
            )
        )
    )
    pyc.exec("add = f[$x + $y]; 1 |> add($, y=42) |> expect[$ == 43]")
    try:
        pyc.exec(
            textwrap.dedent(
                """
                def add(x, y):
                    return x + y
                1 |> add($, y=42) |> expect[$ == 44]
                """.strip(
                    "\n"
                )
            )
        )
    except AssertionError:
        pass  # expected
    else:
        assert False
    try:
        pyc.exec("add = f[$x + $y]; 1 |> add($, y=42) |> expect[$ == 44]")
    except AssertionError:
        pass  # expected
    else:
        assert False
    pyc.exec("42 |> print($, end=' ')")


def test_keyword_placeholder():
    pyc.exec(
        "func = sorted([1, 3, 2], reverse=$); assert func(False) == [1, 2, 3]; func(True) |> expect[$ == [3, 2, 1]]"
    )
    try:
        pyc.exec(
            "func = sorted([1, 3, 2], reverse=$); assert func(False) == [1, 2, 3]; func(True) |> expect[$ == []]"
        )
    except AssertionError:
        pass  # expected
    else:
        assert False


def test_named_placeholders_simple():
    assert pyc.eval("reduce[$x + $y]([1, 2, 3])") == 6
    assert pyc.eval("sorted($lst, reverse=True)([1, 2, 3])") == [3, 2, 1]


def test_named_placeholders_multiple_appearances():
    assert pyc.eval("(2, 3, 4) *|> f[$a*$b + $b*$c + $a*$c]") == 26
    # reset the placeholder counter to trigger a collision if we don't handle explicit digit placeholders properly
    SingletonArgCounterMixin._arg_ctr = 0
    assert pyc.eval("(1, 1, 2, 3, 4) *|> f[$1*$2 + $2*$3 + $1*$3 + $ + $]") == 28


def test_named_placeholders_complex():
    assert (
        pyc.eval(
            "zip(['*', '+', '+'], [[2, 3, 4], [1, 2, 3], [4, 5, 6]]) "
            "|> map[$ *|> reduce({'*': f[$ * $], '+': f[$ + $]}[$op], $row)] "
            "|> sum"
        )
        == 45
    )
    assert (
        pyc.eval(
            "zip(['*', '+', '+'], [[2, 3, 4], [1, 2, 3], [4, 5, 6]]) "
            "|> map[$ *|> reduce({'*': f[$x * $y], '+': f[$x + $y]}[$op], $row)] "
            "|> sum"
        )
        == 45
    )
    assert (
        pyc.eval(
            "zip(['*', '+', '+'], [[2, 3, 4], [1, 2, 3], [4, 5, 6]]) "
            "|> map[$ *|> ($op, $row) *|> reduce({'*': f[$x * $y], '+': f[$x + $y]}[$op], $row)] "
            "|> sum"
        )
        == 45
    )
    assert (
        pyc.eval(
            "zip(['*', '+', '+'], [[2, 3, 4], [1, 2, 3], [4, 5, 6]]) "
            "|> map[$ *|> ($op, $row) *|> reduce({'*': f[$ * $], '+': f[$ + $]}[$op], $row)] "
            "|> sum"
        )
        == 45
    )


def test_dict_operators():
    assert pyc.eval("{'a': 1, 'b': 2} **|> dict") == {"a": 1, "b": 2}
    assert pyc.eval("{'a': 1, 'b': 2} **$> dict <|** {'c': 3, 'd': 4}") == {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
    }
    assert pyc.eval("{'a': 1, 'b': 2} **|> (dict <$** {'c': 3, 'd': 4})") == {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
    }
    assert pyc.eval("[('a',1), ('b', 2)] |> (list <. dict <.** dict)") == [
        "a",
        "b",
    ]
    assert pyc.eval("[('a',1), ('b', 2)] |> (dict **.> dict .> list)") == [
        "a",
        "b",
    ]


def test_multiline_pipeline():
    pyc.exec(
        textwrap.dedent(
            """
            add1 = (
                $
                |> $ + 1
            )
            assert 1 |> add1 == 2
            """.strip(
                "\n"
            )
        )
    )


def test_multiline_pipeline_again():
    assert (
        pyc.eval(
            textwrap.dedent(
                """
            1 |> ($ |>
                $ + 1 |>
                $ + 2 |>
                $ + 3 |>
                when[$ > 3] |>
                $ + 4
            ) .** 3
            """.strip(
                    "\n"
                )
            )
        )
        == 31
    )


def test_multistep_multiline_pipeline():
    pyc.exec(
        textwrap.dedent(
            """
            add_stuff = $ |> $ + 1 |> $ + 2 |> $ + 3
            assert 1 |> add_stuff == 7
            """.strip(
                "\n"
            )
        )
    )
    pyc.exec(
        textwrap.dedent(
            """
            add_stuff = (
                $
                |> $ + 1
                |> $ + 2
                |> $ + 3
            )
            assert 1 |> add_stuff == 7
            """.strip(
                "\n"
            )
        )
    )


def test_comprehension_placeholder():
    assert pyc.eval(
        "'1-2,5-6,3-4'.strip().split(',') "
        "|> [v.strip().split('-') for v in $] "
        "|> [[int(v1), int(v2)] for v1, v2 in $] "
        "|> sorted |> sum($, [])"
    ) == [1, 2, 3, 4, 5, 6]


def test_chain_with_placeholder():
    assert pyc.eval("[3, 2, 1] |> sorted($).index(1)") == 0


def test_immediately_evaluated_placeholder():
    assert pyc.eval("sorted($, reverse=True)([2, 1, 3])") == [3, 2, 1]


def test_quick_maps():
    assert pyc.eval("['1', '2', '3'] |> map[int]") == [1, 2, 3]
    assert pyc.eval("['1', '2', '3'] |> map[int($)]") == [1, 2, 3]
    assert pyc.eval("['1', '2', '3'] |> map[int] |> map[$ % 2==0]") == [
        False,
        True,
        False,
    ]
    assert (
        pyc.eval(
            "zip(['*', '+', '+'], [[2, 3, 4], [1, 2, 3], [4, 5, 6]]) "
            "|> map[$ *|> reduce({'*': f[$ * $], '+': f[$ + $]}[$], $)] "
            "|> sum"
        )
        == 45
    )


def test_pipeline_map_with_quick_lambda_applied():
    assert pyc.eval("[[1, 2], [3, 4]] |> map[f[$ + $](*$)]") == [
        3,
        7,
    ]


def test_quick_reduce():
    assert pyc.eval("reduce[$ + $]([1, 2, 3, 4])") == 10
    assert pyc.eval("reduce[f[$ + $]]([1, 2, 3, 4])") == 10
    assert pyc.eval("reduce[$ + $] <| [1, 2, 3, 4]") == 10
    assert pyc.eval("reduce[f[$ + $]] <| [1, 2, 3, 4]") == 10
    assert pyc.eval("reduce[$ + $ |> $] <| [1, 2, 3, 4]") == 10
    assert pyc.eval("reduce[$ + $ |> 2*$] <| [1, 2, 3, 4]") == 44


def test_quick_filter():
    assert pyc.eval("filter[$ % 2 == 0]([1, 2, 3, 4, 5])") == [2, 4]
    assert pyc.eval("filter[$ % 2 == 1]([1, 2, 3, 4, 5])") == [1, 3, 5]
    assert pyc.eval("filter[$ % 2 == 0](range(5)) |> list") == [0, 2, 4]
    assert pyc.eval("filter[$ % 2 == 1](range(5)) |> list") == [1, 3]


def test_named_unpack():
    assert pyc.eval("'a: b c d' |> $.strip().split(': ') *|> ($, $.split())") == (
        "a",
        ["b", "c", "d"],
    )
    assert pyc.eval(
        "'a: b c d' |> $.strip().split(': ') *|> ($node, $adj.split())"
    ) == ("a", ["b", "c", "d"])
    assert pyc.eval(
        "'a: b c d' |> $.strip().split(': ') *|> ($node, $adj.split()) *|> ($adj, $node)"
    ) == (["b", "c", "d"], "a")


def test_partial_calls():
    assert pyc.eval("[2, 3, 4] |> reduce$(lambda x, y: x * y)") == 24
    assert pyc.eval("[2, 3, 4] |> reduce$(f[$ * $])") == 24
    assert pyc.eval("reduce[$ * $]$([2, 3, 4])()") == 24
    assert pyc.eval("reduce[$ * $]$([2, 3, 4])(2)") == 48
    assert pyc.eval("reduce[$ * $]$()([2, 3, 4])") == 24
    assert pyc.eval("reduce[$ * $]$()([2, 3, 4], 2)") == 48
    assert pyc.eval("[2, 3, 4] |> reduce[$ * $]$()") == 24
    assert pyc.eval("[2, 3, 4] |> reduce[$ * $]($, 2)") == 48
    assert pyc.eval("[2, 3, 4] |> reduce[$ * $]$($, 2)()") == 48


def test_forward_function_composition():
    assert pyc.eval("([1], [2], [3, 4]) |> (list .> sum($, start=[]))") == [
        1,
        2,
        3,
        4,
    ]
    assert pyc.eval("([1], [2], [3, 4]) |> (sum($, start=[]) <. list)") == [
        1,
        2,
        3,
        4,
    ]
    assert pyc.eval(
        "[[[1, 2], [3, 4]], [[5, 6]]] |> (sum($, start=[]) *.> zip .> map[list] .> list)"
    ) == [[1, 3, 5], [2, 4, 6]]


def test_backward_compose_pipeline():
    assert pyc.eval("sum($, start=[]) <. list <| ([1], [2], [3, 4])") == [
        1,
        2,
        3,
        4,
    ]


def test_nullpipe_op():
    assert pyc.eval("1 |> {0: 42, 1: None}[$] ?> $ + 1 ?> $ + 2") is None
    assert pyc.eval("1 |> {0: 42, 1: None}[$] ?> $ + 1 |> $ + 2 |> $ + 3") is None
    assert pyc.eval("0 |> {0: 42, 1: None}[$] ?> $ + 1 |> $ + 2") == 45


def test_tuple_pipeline_lambda():
    pyc.exec(
        textwrap.dedent(
            """
            add = ($x, $y) *|> $x + $y
            assert add(1, 2) == 3
            assert (1, 2) *|> add == 3
            """.strip(
                "\n"
            )
        )
    )


def test_placeholder_arg_ordering():
    assert pyc.eval("(1, 2, 3) *|> ($x, $y, $z) *|> ($y, $z, $x)") == (2, 3, 1)
    assert pyc.eval("($x, $y, $z) *|> ($y, $z, $x) <|* (1, 2, 3)") == (2, 3, 1)
    assert pyc.eval("(1, 2, 3) *|> ($x, $y, $z) *|> ($y, $z, $x) *|> ($z, $x, $y)") == (
        3,
        1,
        2,
    )
    assert pyc.eval("($x, $y, $z) *|> ($y, $z, $x) *|> ($z, $x, $y) <|* (1, 2, 3)") == (
        3,
        1,
        2,
    )


def test_placeholder_scope_within_pipeline_step():
    assert pyc.eval("'12' |> list *|> (int($) - 1, int($) + 1)") == (0, 3)


def test_loop_pipelines():
    pyc.exec(
        textwrap.dedent(
            """
            for _ in range(10):
                assert 1 |> $ + 1 == 2
            """
        ).strip("\n")
    )


def test_function_pipelines():
    pyc.exec(
        textwrap.dedent(
            """
            def function_with_pipeline():
                return 1 |> $ + 1
                
            assert function_with_pipeline() == 2
            """
        ).strip("\n")
    )


def test_do():
    assert pyc.eval("[0, 1, 2] |>> lst |> do[$.append(42)]") == [
        0,
        1,
        2,
        42,
    ]


def test_fork():
    assert pyc.eval("0 |> fork[$+1, $+2]") == (1, 2)
    assert pyc.eval("fork[$+1, $+2](0)") == (1, 2)
    assert pyc.eval("0 |> fork[$|>$+1, $|>$+2]") == (1, 2)
    assert pyc.eval("fork[$|>$+1, $|>$+2](0)") == (1, 2)


def test_map_frozenset_list_set_tuple_eagerness():
    assert pyc.eval("[1, 2, 3] |> map[$ + 1]") == [2, 3, 4]
    assert pyc.eval("(1, 2, 3) |> map[$ + 1]") == (2, 3, 4)
    assert pyc.eval("{1, 2, 3} |> map[$ + 1]") == {2, 3, 4}
    assert pyc.eval("{1, 2, 3} |> frozenset |> map[$ + 1]") == frozenset({2, 3, 4})
    assert pyc.eval("{1: 1, 2: 2, 3: 3} |> map[$ + 1] |> isinstance($, dict)") is False
    assert pyc.eval("{1: 1, 2: 2, 3: 3} |> map[$ + 1] |> list") == [2, 3, 4]


def test_when():
    assert pyc.eval("1 |> when[$ > 0] |> $ + 1") == 2
    assert pyc.eval("1 |> when[$ < 0] |> $ + 1") is None
    assert (
        pyc.eval(
            "1 |> fork[$ |> when[$ < 0] |> $ - 41, $ |> when[$ >= 0] |> $ + 41] |> collapse"
        )
        == 42
    )
    assert (
        pyc.eval("1 |> fork[when[$ < 0] .> $ - 41, when[$ >= 0] .> $ + 41] |> collapse")
        == 42
    )


def test_otherwise():
    assert (
        pyc.eval("1 |> fork[$ |> when[$ < 0] |> $ - 41, otherwise[$ + 41]] |> collapse")
        == 42
    )
    assert (
        pyc.eval("1 |> fork[when[$ < 0] .> $ - 41, otherwise[$ + 41]] |> collapse")
        == 42
    )
    assert (
        pyc.eval("1 |> fork[$ |> when[$ < 0] |> $ - 41, otherwise[$ + 41]] |> collapse")
        == 42
    )
    assert (
        pyc.eval(
            "1 |> fork[$ |> when[$ >= 0] |> $ + 41, otherwise[$ - 41]] |> collapse"
        )
        == 42
    )
    assert (
        pyc.eval("1 |> fork[when[$ >= 0] .> $ + 41, otherwise[$ - 41]] |> collapse")
        == 42
    )


def test_nested_otherwise():
    pyc.exec(
        textwrap.dedent(
            """
            threshold = $ |> fork[
                when[$ > 42] .> replace(42),
                otherwise[fork[
                    when[$ > 20] .> replace(20),
                    otherwise[fork[
                        when[$ > 10] .> replace(10),
                        otherwise[replace(0)],
                    ] .> collapse]
                ] .> collapse],
            ] |> collapse
            
            1 |> threshold |> expect[$ == 0]
            5 |> threshold |> expect[$ == 0]
            11 |> threshold |> expect[$ == 10]
            15 |> threshold |> expect[$ == 10]
            21 |> threshold |> expect[$ == 20]
            31 |> threshold |> expect[$ == 20]
            41 |> threshold |> expect[$ == 20]
            44 |> threshold |> expect[$ == 42]
            99 |> threshold |> expect[$ == 42]
            """.strip(
                "\n"
            )
        )
    )
    pyc.exec(
        textwrap.dedent(
            """
            pthreshold = $ |> parallel[
                when[$ > 42] .> replace(42),
                otherwise[fork[
                    when[$ > 20] .> replace(20),
                    otherwise[fork[
                        when[$ > 10] .> replace(10),
                        otherwise[replace(0)],
                    ] .> collapse]
                ] .> collapse],
            ] |> collapse

            1 |> pthreshold |> expect[$ == 0]
            5 |> pthreshold |> expect[$ == 0]
            11 |> pthreshold |> expect[$ == 10]
            15 |> pthreshold |> expect[$ == 10]
            21 |> pthreshold |> expect[$ == 20]
            31 |> pthreshold |> expect[$ == 20]
            41 |> pthreshold |> expect[$ == 20]
            44 |> pthreshold |> expect[$ == 42]
            99 |> pthreshold |> expect[$ == 42]
            """.strip(
                "\n"
            )
        )
    )


def test_future():
    assert pyc.eval("1 |> future[$ + 1] |> $.result()") == 2
    assert pyc.eval("[1, 2, 3] |> future[sum] |> $.result()") == 6


def test_parallel():
    assert pyc.eval("[1, 2, 3, 4] |> parallel[sum, reduce[$*$]]") == (
        10,
        24,
    )


def test_do_multiple():
    pyc.exec(
        textwrap.dedent(
            """
            lst = []
            def func(*args):
                lst.extend(args)
            (1, 2) *|> do[func] |> null
            assert lst == [1, 2]
            """.strip(
                "\n"
            )
        )
    )
    pyc.exec(
        textwrap.dedent(
            """
            lst = []
            (1, 2) *|> do[lst.extend([$, $])] |> null
            assert lst == [1, 2]
            """.strip(
                "\n"
            )
        )
    )


def test_dict_pipeline_operator_with_named_args():
    assert pyc.eval("{'x': 1, 'y': 2} **|> f[$x + $y]") == 3
    assert pyc.eval("{'x': 1, 'y': 2} **|> $x + $y") == 3


def test_environment_init():
    env = pyc.exec("def f(x, y): return x + y")
    with patch.object(
        pipescript.utils, pipescript.utils._get_user_ns_impl.__name__, return_value=env
    ):
        PipelineTracer.instance().reset()
        MacroTracer.instance().reset()
        OptionalChainingTracer.instance().reset()
        assert pyc.eval("(1, 2) *|> f", global_env=env) == 3


def test_partial_conflict_with_placeholder():
    assert pyc.eval("[1, 2, 3] |> (type($v), reversed($v)) *|> $($)") == [
        3,
        2,
        1,
    ]
    # TODO: this requires smarter augmentation specs that can match regular expressions
    # assert pyc.eval("[1, 2, 3] |> (type($v), reversed($v)) *|>$($)") == [3, 2, 1]


def test_function_exponentiation_and_repeat():
    pyc.exec(
        textwrap.dedent(
            """
            collatz_vals = []
            collatz = when[$ != 1] .> fork[
                when[$ % 2 == 0] .> $ // 2,
                when[$ % 2 == 1] .> $ * 3 + 1,
            ] .> collapse .> do[collatz_vals.append($)]
            42 |> collatz .** 100 |> null
            assert collatz_vals == [21, 64, 32, 16, 8, 4, 2, 1]
            collatz_vals.clear()
            42 |> repeat[collatz] |> null
            assert collatz_vals == [21, 64, 32, 16, 8, 4, 2, 1]
            """.strip(
                "\n"
            )
        )
    )
    assert pyc.eval("($ |> $ + 1) .** 3 <| 1") == 4
    assert pyc.eval("1 |> ($ |> $ + 1) .** 3") == 4
    assert pyc.eval("f[$ + $]($, 1) .** 3 <| 1") == 4
    assert pyc.eval("1 |> f[f[$ + $]($, 1)] .** 3") == 4


def test_multi_arg_function_exponentiation():
    pyc.exec(
        textwrap.dedent(
            """
            def square(v1, v2, v3):
                return v1**2, v2**2, v3**2
                
            assert (1, 2, 3) *|> square .** 2 == (1, 16, 81)
            
            def triple(*args):
                return tuple(3*v for v in args)
                
            assert (1, 2, 3) *|> triple .** 2 == (9, 18, 27)
            """.strip(
                "\n"
            )
        )
    )


def test_repeat_until():
    assert (
        pyc.eval(
            textwrap.dedent(
                """
            [42] |> repeat[
                until[$[-1] == 1] .> fork[
                    when[$[-1]%2 == 0] .> $v + [$v[-1]//2],
                    unless[$[-1]%2 == 0] .> $v + [$v[-1]*3 + 1],
                ] .> collapse
            ]
            """.strip(
                    "\n"
                )
            )
        )
        == [42, 21, 64, 32, 16, 8, 4, 2, 1]
    )


def test_repeat_until_fancy():
    assert (
        pyc.eval(
            textwrap.dedent(
                """
            [42] |> repeat[
                until[$[-1] == 1] .> fork[
                    $,
                    $[-1] |> fork[
                        when[$ % 2 == 0] .> $ // 2,
                        unless[$ % 2 == 0] .> $ * 3 + 1,
                    ] |> collapse,
                ] *.> $ + [$]
            ]
            """.strip(
                    "\n"
                )
            )
        )
        == [42, 21, 64, 32, 16, 8, 4, 2, 1]
    )


def test_shift():
    assert pyc.eval("(1, 2, 3) |> lshift") == (2, 3, 1)
    assert pyc.eval("(1, 2, 3) |> rshift") == (3, 1, 2)
    assert pyc.eval("(1, 2, 3) |> fork[rshift .> rshift, lshift] *|> ($ == $)")
    assert pyc.eval("(1, 2, 3) |> fork[lshift .> lshift, rshift] *|> ($ == $)")


def test_repeat_until_fancy_push_pop_shift():
    assert (
        pyc.eval(
            textwrap.dedent(
                """
            [42] |> repeat[
                until[$[-1] == 1] .> push .> $[-1] .> fork[
                    when[$ % 2 == 0] .> $ // 2,
                    unless[$ % 2 == 0] .> $ * 3 + 1,
                ] .> collapse .> pop .> lshift *.> $ + [$]
            ]
            """.strip(
                    "\n"
                )
            )
        )
        == [42, 21, 64, 32, 16, 8, 4, 2, 1]
    )


def test_fancy_fact_table():
    pyc.exec(
        textwrap.dedent(
            """
            make_fact_table = $ |> ([1], $) *|> repeat[until[len($)==$+1] .> do[$[0] |> $v.append($v[-1]*len($v))]] |> $[0]
            
            4 |> make_fact_table |> expect[$ == [1, 1, 2, 6, 24]]
            5 |> make_fact_table |> expect[$ == [1, 1, 2, 6, 24, 120]]
            """.strip(
                "\n"
            )
        )
    )


def test_repeat_ntimes():
    assert pyc.eval("[] |> repeat[ntimes[5] .> do[$.append(42)]]") == [42] * 5
    assert pyc.eval("[] |> repeat[ntimes[2 + 2] .> do[$.append(42)]]") == [42] * 4
    assert pyc.eval("[] |> repeat[ntimes[2 |> $ + 5] .> do[$.append(42)]]") == [42] * 7


def test_read_write():
    assert pyc.eval("42 |> write$('x') |> $ + 1 |> read$('x')") == (43, 42)


def test_read_write_macros():
    assert pyc.eval("42 |> write[x] |> $ + 1 |> read[x]") == (43, 42)


def test_unnest():
    assert pyc.eval("42 |> push |> push |> pop |> pop |> unnest") == (42, 42, 42)


def test_context_and_expect():
    pyc.exec(
        textwrap.dedent(
            """
            from contextlib import contextmanager
            
            stack = []
            
            @contextmanager
            def add_stack(val):
                stack.append(val)
                yield stack
                stack.pop()
            
            42 |> add_stack |> context[expect[len($) == 1]] |> expect[len($) == 0]
            42 |> add_stack($) |> context[expect[len($) == 1]] |> expect[len($) == 0]
            42 |> add_stack |> context[expect[lambda *_: len(stack) == 1]] |> expect[lambda *_: len(stack) == 0]
            42 |> add_stack($) |> context[expect[lambda *_: len(stack) == 1]] |> expect[lambda *_: len(stack) == 0]
            
            try:
                42 |> add_stack |> context[expect[len($) == 0]] |> expect[len($) == 0]
            except AssertionError:
                pass
            else:
                assert False
            
            try:
                42 |> add_stack |> context[expect[len($) == 1]] |> expect[len($) == 1]
            except AssertionError:
                pass
            else:
                assert False
            """.strip(
                "\n"
            )
        )
    )


def test_replace():
    assert pyc.eval("42 |> replace(0)") == 0
    assert pyc.eval("0 |> replace(42)") == 42


def test_once():
    pyc.exec(
        textwrap.dedent(
            """
            from pipescript.constants import pipeline_null
            
            def countdown(x):
                def passes(obj):
                    nonlocal x
                    if x <= 0:
                        return pipeline_null
                    x -= 1
                    return obj
                return passes
            
            # without the `once` macro, the repeat loop would run infinitely
            assert [] |> repeat[once[countdown(5)] .> do[$.append(42)]] == [42] * 5
            """.strip(
                "\n"
            )
        )
    )


def test_memoize():
    pyc.exec(
        textwrap.dedent(
            """
            import random
            
            func = memoize[random.random]
            assert func() == func()
            
            add1 = memoize[$ + 1]
            assert 1 |> add1 == 2
            assert 2 |> add1 == 3
            
            add42 = memoize[$ |> $ + 42]
            assert 1 |> add42 == 43
            assert 2 |> add42 == 44
            """.strip(
                "\n"
            )
        )
    )


def test_recursion():
    pyc.exec(
        textwrap.dedent(
            """
            fact = $ |> fork[when[$ <= 1] .> replace(1), when[$ > 1] .> $v * fact($v-1)] |> collapse
            
            assert fact(3) == 6
            assert fact(4) == 24
            assert fact(5) == 120
            """
        )
    )
    pyc.exec(
        textwrap.dedent(
            """
            fact = $ |> fork[when[$ <= 1] .> replace(1), otherwise[$v * fact($v-1)]] |> collapse

            3 |> fact |> expect[$ == 6]
            4 |> fact |> expect[$ == 24]
            5 |> fact |> expect[$ == 120]
            """
        )
    )
    pyc.exec(
        textwrap.dedent(
            """
            fact = $ |> memoize[fork[when[$ <= 1] .> replace(1), otherwise[$v * fact($v-1)]] .> collapse]

            3 |> fact |> expect[$ == 6]
            4 |> fact |> expect[$ == 24]
            5 |> fact |> expect[$ == 120]
            3 |> fact |> expect[$ == 6]
            4 |> fact |> expect[$ == 24]
            5 |> fact |> expect[$ == 120]
            """
        )
    )
    pyc.exec(
        textwrap.dedent(
            """
            fact = memoize[$ |> fork[when[$ <= 1] .> replace(1), otherwise[$v * fact($v-1)]] |> collapse]

            3 |> fact |> expect[$ == 6]
            4 |> fact |> expect[$ == 24]
            5 |> fact |> expect[$ == 120]
            3 |> fact |> expect[$ == 6]
            4 |> fact |> expect[$ == 24]
            5 |> fact |> expect[$ == 120]
            """
        )
    )
    pyc.exec(
        textwrap.dedent(
            """
            fact = $ |> parallel[when[$ <= 1] .> replace(1), otherwise[$v * fact($v-1)]] |> collapse

            3 |> fact |> expect[$ == 6]
            4 |> fact |> expect[$ == 24]
            5 |> fact |> expect[$ == 120]
            """
        )
    )
