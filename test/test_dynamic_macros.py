from __future__ import annotations

from typing import Any, Generator

import pyccolo as pyc
import pytest

from pipescript.tracers.macro_tracer import DynamicMacro, MacroTracer
from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer
from pipescript.tracers.pipeline_tracer import PipelineTracer


@pytest.fixture(autouse=True)
def all_tracers() -> Generator[None, None, None]:
    with PipelineTracer:
        with MacroTracer:
            with OptionalChainingTracer:
                yield


def refresh_dynamic_macros(env: dict[str, Any]) -> None:
    MacroTracer.dynamic_macros.clear()
    for k, v in env.items():
        if isinstance(v, DynamicMacro):
            MacroTracer.dynamic_macros[k] = v
    MacroTracer.instance().reset()


def test_simple_dynamic_macro():
    refresh_dynamic_macros(pyc.exec("switch = macro[fork[$] .> collapse]"))
    assert pyc.eval("1 |> switch[when[$==0] .> $+1, when[$==1] .> $-1]") == 0
    assert pyc.eval("0 |> switch[when[$==0] .> $+1, when[$==1] .> $-1]") == 1


def test_pipeline_placeholder_dynamic_macro():
    refresh_dynamic_macros(pyc.exec("switch = macro[$ |> fork[$] |> collapse]"))
    assert pyc.eval("1 |> switch[when[$==0] .> $+1, when[$==1] .> $-1]") == 0
    assert pyc.eval("0 |> switch[when[$==0] .> $+1, when[$==1] .> $-1]") == 1
