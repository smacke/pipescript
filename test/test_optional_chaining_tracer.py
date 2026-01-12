from __future__ import annotations

import textwrap
from typing import Generator

import pyccolo as pyc
import pytest

from pipescript.tracers.macro_tracer import MacroTracer
from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer
from pipescript.tracers.pipeline_tracer import PipelineTracer


@pytest.fixture(autouse=True)
def all_tracers() -> Generator[None, None, None]:
    with PipelineTracer:
        with MacroTracer:
            with OptionalChainingTracer:
                yield


def test_optional_chaining_simple():
    pyc.exec(
        textwrap.dedent(
            """
            class Foo:
                def __init__(self, x):
                    self.x = x
            foo = Foo(Foo(Foo(None)))
            try:
                bar = foo.x.x.x.x
            except:
                pass
            else:
                assert False
            assert foo.x.x.x?.x is None
            assert foo.x.x.x?.x() is None
            assert foo.x.x.x?.x?.whatever is None
            assert isinstance(foo?.x?.x, Foo)
            assert isinstance(foo.x?.x, Foo)
            assert isinstance(foo?.x.x, Foo)
            """.strip(
                "\n"
            )
        )
    )


def test_permissive_attr_vs_optional_attr_qualifier():
    try:
        pyc.exec("foo = object(); assert foo?.bar is None")
    except AttributeError:
        pass
    else:
        assert False

    try:
        pyc.exec("foo = object(); assert foo.?bar.baz is None")
    except AttributeError:
        pass
    else:
        assert False

    pyc.exec("foo = object(); assert foo.?bar is None")

    pyc.exec("foo = object(); assert foo.?bar?.baz is None")

    pyc.exec("foo = object(); assert foo.?bar?.baz.bam() is None")


def test_call_on_optional():
    pyc.exec("foo = None; assert foo?.() is None")


def test_nullish_coalescing():
    pyc.exec("None ?? None")
    pyc.exec("foo = ''; assert (foo ?? None) == ''")
    pyc.exec("foo = ''; assert (foo       ?? None) == ''")
    pyc.exec("foo = ''; assert (foo       ??       None) == ''")
    pyc.exec("foo = ''; assert (foo ??       None) == ''")
    assert pyc.eval("'' ?? None") == ""
    assert pyc.eval("''??None") == ""
    assert pyc.eval("0 ?? None") == 0
    assert pyc.eval("None ?? 0 ?? None") == 0
    assert pyc.eval("None or 0 ?? None") == 0
    assert pyc.eval("None and 0 ?? None") is None
    assert pyc.eval("0 or None ?? False") is False


def test_multiline_nullish_coalescing():
    assert (
        pyc.eval(
            textwrap.dedent(
                """
            (
                ""
                ??
                None
            )
            """.strip(
                    "\n"
                )
            )
        )
        == ""
    )


def test_everything_everywhere_all_at_once():
    pyc.exec(
        textwrap.dedent(
            """
            d1 = None
            try:
                d1.?["foo"]
            except TypeError:
                pass
            else:
                assert False
            assert d1?.["foo"] is None
            d1 |> expect[$?.["foo"] is None]
            assert d1?.["foo"].bar().baz is None
            d1 |> expect[$?.["foo"].bar().baz is None]
            assert d1?.?["foo"] is None
            d1 |> expect[$?.?["foo"] is None]
            assert d1?.?["foo"].bar().baz is None
            d1 |> expect[$?.?["foo"].bar().baz is None]
            
            d2 = {"bar": 42}
            try:
                d2?.["foo"]
            except KeyError:
                pass
            else:
                assert False
            try:
                d2 |> $?.["foo"]
            except KeyError:
                pass
            else:
                assert False
            assert d2.?["foo"] is None
            d2 |> expect[$.?["foo"] is None]
            try:
                assert d2.?["foo"].bar is None
            except AttributeError:
                pass
            else:
                assert False
            try:
                d2 |> expect[$.?["foo"].bar is None]
            except AttributeError:
                pass
            else:
                assert False
            try:
                assert d2?.?["foo"].bar is None
            except AttributeError:
                pass
            else:
                assert False
            try:
                d2 |> expect[$?.?["foo"].bar is None]
            except AttributeError:
                pass
            else:
                assert False
            try:
                assert d2?.?["foo"].bar.baz(bam().bat().zzzz).yyyy is None
            except AttributeError:
                pass
            else:
                assert False
            try:
                d2 |> expect[$?.?["foo"].bar.baz(bam().bat().zzzz).yyyy is None]
            except AttributeError:
                pass
            else:
                assert False
            assert d2.?["foo"]?.bar is None
            d2 |> expect[$.?["foo"]?.bar is None]
            assert d2.?["foo"]?.bar.baz().yyyy is None
            d2 |> expect[$.?["foo"]?.bar.baz().yyyy is None]
            assert d2.?["foo"]?.bar.baz(bam().bat).yyyy is None
            d2 |> expect[$.?["foo"]?.bar.baz(bam().bat).yyyy is None]
            assert d2.?["foo"]?.bar.baz(d1().bat).yyyy is None
            d2 |> expect[$.?["foo"]?.bar.baz(d1().bat).yyyy is None]
            assert d2?.?["foo"] is None
            d2 |> expect[$?.?["foo"] is None]
            assert d2?.?["foo"]?.bar is None
            d2 |> expect[$?.?["foo"]?.bar is None]
            assert d2?.?["foo"]?.bar.baz(bam().bat().zzzz).yyyy is None
            d2 |> expect[$?.?["foo"]?.bar.baz(bam().bat().zzzz).yyyy is None]
            assert d2?.?["foo"]?.bar.baz(d1().bat().zzzz).yyyy is None
            d2 |> expect[$?.?["foo"]?.bar.baz(d1().bat().zzzz).yyyy is None]
            assert d2?.?["foo"]?.bar.baz(d1).yyyy is None
            d2 |> expect[$?.?["foo"]?.bar.baz(d1).yyyy is None]

            assert d2["bar"] == 42
            d2 |> expect[$["bar"] == 42]
            assert d2?.["bar"] == 42
            d2 |> expect[$?.["bar"] == 42]
            assert d2.?["bar"] == 42
            d2 |> expect[$.?["bar"] == 42]
            assert d2?.?["bar"] == 42
            d2 |> expect[$?.?["bar"] == 42]
            """.strip(
                "\n"
            )
        )
    )
