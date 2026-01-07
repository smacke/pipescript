from __future__ import annotations

import textwrap

import pyccolo as pyc

from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer


def test_optional_chaining_simple():
    with OptionalChainingTracer:
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
    with OptionalChainingTracer:
        try:
            pyc.exec("foo = object(); assert foo?.bar is None")
        except AttributeError:
            pass
        else:
            assert False

    with OptionalChainingTracer:
        try:
            pyc.exec("foo = object(); assert foo.?bar.baz is None")
        except AttributeError:
            pass
        else:
            assert False

    with OptionalChainingTracer:
        pyc.exec("foo = object(); assert foo.?bar is None")

    with OptionalChainingTracer:
        pyc.exec("foo = object(); assert foo.?bar?.baz is None")

    with OptionalChainingTracer:
        pyc.exec("foo = object(); assert foo.?bar?.baz.bam() is None")


def test_call_on_optional():
    with OptionalChainingTracer:
        pyc.exec("foo = None; assert foo?.() is None")


def test_nullish_coalescing():
    with OptionalChainingTracer:
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
    with OptionalChainingTracer:
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
