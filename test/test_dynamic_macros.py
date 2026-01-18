from __future__ import annotations

import textwrap
from typing import Any, Generator
from unittest.mock import patch

import pyccolo as pyc
import pytest

import pipescript.utils
from pipescript.tracers.macro_tracer import DynamicMacro, MacroTracer
from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer
from pipescript.tracers.pipeline_tracer import PipelineTracer


@pytest.fixture(autouse=True)
def all_tracers() -> Generator[None, None, None]:
    MacroTracer.dynamic_macros.clear()
    with PipelineTracer:
        with MacroTracer:
            with OptionalChainingTracer:
                yield


def refresh_dynamic_macros(env: dict[str, Any]) -> None:
    for k, v in env.items():
        if not isinstance(v, DynamicMacro):
            continue
        if v.is_method:
            MacroTracer.dynamic_method_macros[k] = v
        else:
            MacroTracer.dynamic_macros[k] = v
    MacroTracer.instance().reset()


def test_simple_dynamic_macro():
    refresh_dynamic_macros(pyc.exec("switch = macro[fork[$$] .> collapse]"))
    assert pyc.eval("1 |> switch[when[$==0] .> $+1, when[$==1] .> $-1]") == 0
    assert pyc.eval("0 |> switch[when[$==0] .> $+1, when[$==1] .> $-1]") == 1


def test_simple_dynamic_macro_named_arg():
    refresh_dynamic_macros(pyc.exec("switch = macro[fork[$$v] .> collapse]"))
    assert pyc.eval("1 |> switch[when[$==0] .> $+1, when[$==1] .> $-1]") == 0
    assert pyc.eval("0 |> switch[when[$==0] .> $+1, when[$==1] .> $-1]") == 1


def test_pipeline_placeholder_dynamic_macro():
    refresh_dynamic_macros(pyc.exec("switch = macro[$ |> fork[$$] |> collapse]"))
    assert pyc.eval("1 |> switch[when[$==0] .> $+1, when[$==1] .> $-1]") == 0
    assert pyc.eval("0 |> switch[when[$==0] .> $+1, when[$==1] .> $-1]") == 1


def test_recursive_macro_expansion():
    refresh_dynamic_macros(pyc.exec("switch = macro[fork[$$] .> collapse]"))
    refresh_dynamic_macros(
        pyc.exec("flip = macro[$$ |> switch[when[$==0] .> $+1, when[$==1] .> $-1]]")
    )
    assert pyc.eval("flip[0]") == 1
    assert pyc.eval("flip[1]") == 0


def test_pipeline_recursive_macro_expansion():
    refresh_dynamic_macros(pyc.exec("switch = macro[$ |> fork[$$] |> collapse]"))
    refresh_dynamic_macros(
        pyc.exec("flip = macro[$$ |> switch[when[$==0] .> $+1, when[$==1] .> $-1]]")
    )
    assert pyc.eval("flip[0]") == 1
    assert pyc.eval("flip[1]") == 0


def test_macro_arg_order():
    refresh_dynamic_macros(pyc.exec("flip = macro[($$b, $$a), a, b]"))
    assert pyc.eval("flip[0, 1]") == (1, 0)
    assert pyc.eval("flip[1, 0]") == (0, 1)


def test_macro_with_callable():
    env = pyc.exec(
        textwrap.dedent(
            """
            import ast

            class BinaryOpExtractor(ast.NodeVisitor):
                def __init__(self):
                    self.op = None

                def visit_BinOp(self, node):
                    self.op = node.op

                def __call__(self, node):
                    self.visit(node)
                    if self.op is None:
                        raise ValueError("no unary op in expr")
                    elif isinstance(self.op, ast.Add):
                        return f[$ + $]
                    elif isinstance(self.op, ast.Sub):
                        return f[$ - $]
                    elif isinstance(self.op, ast.Mult):
                        return f[$ * $]
                    elif isinstance(self.op, ast.Div):
                        return f[$ / $]
            """.strip(
                "\n"
            )
        ),
    )
    with patch.object(
        pipescript.utils, pipescript.utils._get_user_ns_impl.__name__, return_value=env
    ):
        refresh_dynamic_macros(pyc.exec("opf = macro[BinaryOpExtractor]"))
    assert pyc.eval("op[+](1, 1)") == 2


def test_local_variable_references_in_dynamic_macro_body_expanding_to_pipeline():
    env = pyc.exec("add_x = macro[$$ |> $ + x]\nx=42\nz=2")
    refresh_dynamic_macros(env)
    assert pyc.eval("add_x[1]", global_env=env, local_env=env) == 43
    assert pyc.eval("add_x[x]", global_env=env, local_env=env) == 84
    assert pyc.eval("add_x[z]", global_env=env, local_env=env) == 44
    env = pyc.exec("def foo(y):\n    return add_x[y]", global_env=env, local_env=env)
    assert pyc.eval("foo(1)", global_env=env, local_env=env) == 43
    assert pyc.eval("foo(1)", global_env=env, local_env=env) == 43
    env = pyc.exec("bar = macro[foo($$)]", global_env=env, local_env=env)
    refresh_dynamic_macros(env)
    assert pyc.eval("bar[1]", global_env=env, local_env=env) == 43
    assert pyc.eval("bar[1]", global_env=env, local_env=env) == 43
    env = pyc.exec(
        "add_x_z = macro[z |> add_x[$] |> $ + $$]", global_env=env, local_env=env
    )
    refresh_dynamic_macros(env)
    assert pyc.eval("add_x_z[1]", global_env=env, local_env=env) == 45


def test_macros_are_run_once():
    env = pyc.exec("foreach = macro[$$ |> map[do[$$]] |> list |> null]\nlst=[]")
    refresh_dynamic_macros(env)
    env = pyc.exec("foreach[range(10), lst.append($)]", global_env=env, local_env=env)
    assert pyc.eval("lst", global_env=env, local_env=env) == list(range(10))


def test_simple_method_macro():
    env = pyc.exec("foreach = method[$$ |> map[do[$$]] |> list |> null]\nlst=[]")
    refresh_dynamic_macros(env)
    env = pyc.exec("range(10).foreach[lst.append]", global_env=env, local_env=env)
    assert pyc.eval("lst", global_env=env, local_env=env) == list(range(10))
    env = pyc.exec("range(10).foreach[lst.append($)]", global_env=env, local_env=env)
    assert pyc.eval("lst", global_env=env, local_env=env) == list(range(10)) * 2
    env = pyc.exec("range(10).foreach[$ |> lst.append]", global_env=env, local_env=env)
    assert pyc.eval("lst", global_env=env, local_env=env) == list(range(10)) * 3
    env = pyc.exec(
        "range(10).foreach[$ |> lst.append($)]", global_env=env, local_env=env
    )
    assert pyc.eval("lst", global_env=env, local_env=env) == list(range(10)) * 4
