from __future__ import annotations

from typing import Any, cast

import pyccolo as pyc
import pytest

from pipescript.analysis.pipeline_lowering import lower_pipelines
from pipescript.patches import completion_patch as cp
from pipescript.patches.completion_patch import patch_completer, unpatch_completer
from pipescript.tracers.brace_block_tracer import BraceBlockTracer
from pipescript.tracers.macro_tracer import MacroTracer
from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer
from pipescript.tracers.pipeline_tracer import PipelineTracer


def _tracers() -> list[pyc.BaseTracer]:
    return [
        cast(pyc.BaseTracer, cls).instance()
        for cls in [
            BraceBlockTracer,
            PipelineTracer,
            MacroTracer,
            OptionalChainingTracer,
        ]
    ]


class FakeCompleter:
    def completions(self, text: str, offset: int) -> list[Any]:
        return [(text, offset)]


class ShellNoKernel:
    def __init__(self) -> None:
        self.Completer = FakeCompleter()


class KernelWithDoComplete:
    def do_complete(self, code: str, cursor_pos: int) -> dict[str, Any]:
        return {
            "matches": [],
            "cursor_start": 0,
            "cursor_end": cursor_pos,
            "metadata": {},
            "status": "ok",
        }


class PyodideLikeKernel:
    """Mimics jupyterlite's ``PyodideKernel``: no ``do_complete`` anywhere in
    its MRO, so ``patch_kernel_completer`` must decline to patch it."""


class ShellWithKernel:
    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel
        self.Completer = FakeCompleter()


def test_no_kernel_patches_shell_completer() -> None:
    shell = ShellNoKernel()
    orig_cls = shell.Completer.__class__
    patch_completer(shell, _tracers())
    try:
        assert cp._did_patch_shell_completer is True
        assert shell.Completer.__class__ is not orig_cls
        # the wrapped completer still delegates through to the base impl
        assert shell.Completer.completions("x", 1) == [("x", 1)]
    finally:
        unpatch_completer(shell)
    assert shell.Completer.__class__ is orig_cls


def test_kernel_with_do_complete_is_wrapped() -> None:
    shell = ShellWithKernel(KernelWithDoComplete())
    orig_do_complete = KernelWithDoComplete.do_complete
    patch_completer(shell, _tracers())
    try:
        assert cp._did_patch_shell_completer is False
        assert cp.do_complete_patch_cls is KernelWithDoComplete
        assert KernelWithDoComplete.do_complete is not orig_do_complete
        # shell completer left untouched on the kernel path
        assert shell.Completer.__class__ is FakeCompleter
    finally:
        unpatch_completer(shell)
    assert KernelWithDoComplete.do_complete is orig_do_complete
    assert cp.do_complete_patch_cls is None


def test_pyodide_like_kernel_falls_back_to_shell_completer() -> None:
    shell = ShellWithKernel(PyodideLikeKernel())
    orig_cls = shell.Completer.__class__
    # must not raise AttributeError on object.do_complete
    patch_completer(shell, _tracers())
    try:
        assert cp._did_patch_shell_completer is True
        assert shell.Completer.__class__ is not orig_cls
        assert shell.Completer.completions("x", 1) == [("x", 1)]
    finally:
        unpatch_completer(shell)
    assert shell.Completer.__class__ is orig_cls


LOWERINGS = [
    (
        '["a", "b", "c"] |> "\\n".join($) |> $.up',
        '("\\n".join((["a", "b", "c"]))).up',
    ),
    ("[1, 2] |> sum |> $.bit_", "((sum)(([1, 2]))).bit_"),
    ('{"a": 1} **|> dict |> $.ke', '((dict)(**({"a": 1}))).ke'),
    ("xs ?> len |> $.bit_", "((len)((xs))).bit_"),
    ("[[1], [2]] *|> zip |> list |> $.", "((list)(((zip)(*([[1], [2]])))))."),
    # `<|` applies the accumulated function to the stage
    ('f <| ["a"] |> $.', '(((f))(["a"])).'),
    # `|>>` binds a name and passes its LHS through untouched -- exactly a walrus,
    # which also lets the bound name be typed downstream
    ("xs |>> ys |> $.", "(ys := (xs))."),
    # the cursor may sit *deeper* than the chain it belongs to
    ("foo |> bar($, baz.", "bar((foo), baz."),
    # `|` binds tighter than `==`, so the chain starts after it
    ("a == b |> f |> $.", "a == ((f)((b)))."),
    # a suite-indented chain keeps its indentation
    ('if x:\n    ["a"] |> len |> $.', 'if x:\n    ((len)((["a"]))).'),
    # a named placeholder is one argument no matter how often it repeats
    ('["a"] |> f($v, $v) |> $.', '(f((["a"]), (["a"]))).'),
    # `>>` binds tighter than `|`, so it stays inside the seed
    ("x >> y |> $.z", "(x >> y).z"),
]


@pytest.mark.parametrize("code,expected", LOWERINGS)
def test_lower_pipelines(code: str, expected: str) -> None:
    assert lower_pipelines(code) == expected


# Chains that do not touch the cursor are complete expressions, so they lower to
# values -- which is what lets a name a pipeline bound get typed further down.
WHOLE_CELL_LOWERINGS = [
    (
        'result = ["a"] |> "\\n".join($)\nresult.up',
        'result = "\\n".join((["a"]))\nresult.up',
    ),
    (
        "xs = [1, 2] |> sorted\nys = xs |> len\nys.bit_",
        "xs = (sorted)(([1, 2]))\nys = (len)((xs))\nys.bit_",
    ),
    # `|>>` binds its name, so the walrus makes it visible downstream too
    ("[1, 2] |>> nums |> sum\nnums.", "(sum)((nums := ([1, 2])))\nnums."),
    # a chain nested inside a seed is lowered first, which is what unblocks the
    # chain enclosing it
    ('(["a"] |> len) |> $.', '(((len)((["a"])))).'),
    ("f([1, 2] |> sum) |> $.", "(f((sum)(([1, 2]))))."),
    # a complete chain elsewhere on the cursor's line
    ("f(xs |> g, ba", "f((g)((xs)), ba"),
]


@pytest.mark.parametrize("code,expected", WHOLE_CELL_LOWERINGS)
def test_lower_pipelines_whole_cell(code: str, expected: str) -> None:
    assert lower_pipelines(code) == expected


def test_lower_pipelines_is_linear_in_cell_size() -> None:
    """Foldable chains never nest, so a round lowers all of them in one sweep.
    Re-scanning per chain instead would make each keystroke quadratic in a cell's
    chain count -- 400 chains took over a second before this was batched."""
    import time

    def elapsed(n_chains: int) -> float:
        code = (
            "\n".join(f"v{i} = [1, 2] |> sorted |> $[0]" for i in range(n_chains))
            + "\nv0.bit_"
        )
        lower_pipelines(code)  # warm any lazily-built tables
        start = time.perf_counter()
        lower_pipelines(code)
        return time.perf_counter() - start

    small, large = elapsed(20), elapsed(320)
    # 16x the chains; linear would be ~16x the time. Allow generous slack for a
    # noisy CI box, but not the ~256x that a per-chain rescan would cost.
    assert large < small * 60, (small, large)


def test_lower_pipelines_without_cursor() -> None:
    """Source with no cursor in it (e.g. an LSP's context lines) has no partially
    typed final stage, so a chain running to its end lowers like any other."""
    assert lower_pipelines("x |> f", cursor_at_end=False) == "(f)((x))"
    # ...whereas with a cursor at the end, that same chain is mid-edit
    assert lower_pipelines("x |> f") is None


# A macro's subscript is its own scope, so the `$` inside it belongs to the lambda
# the macro induces. The macros whose effect on the piped value is known get folded;
# the rest still bail. `test_macro_lowerings_match_the_runtime` pins the semantics.
MACRO_LOWERINGS = [
    # `do` runs its body for effect and the guards gate on it, so both hand the
    # piped value straight back -- body and all, the type cannot change
    ('["a"] |> do[print($)] |> $.', '((["a"])).'),
    ('["a"] |> do{ print($) } |> $.', '((["a"])).'),
    ('["a"] |> when[len($) > 0] |> $.', '((["a"])).'),
    # a quick lambda in stage position is just applied to the piped value
    ("[1] |> f[$ + [2]] |> $.", "((([1]) + [2]))."),
    # the lazy variants yield the bare iterator
    ("[1] |> imap[$ * 2] |> $.", "(map((lambda __ps_e: __ps_e * 2), ([1])))."),
    ("[1] |> ifilter[$ > 0] |> $.", "(filter((lambda __ps_e: __ps_e > 0), ([1])))."),
    # a macro nested *inside* a stage is a scope, not a stage head: the stage's own
    # placeholder is the first `$`, and `f[...]`'s is left alone
    (
        '[(1, "b")] |> sorted($, key=f[$[1]]) |> $[0].',
        '(sorted(([(1, "b")]), key=f[$[1]]))[0].',
    ),
]

# The eager `map`/`filter` restore the input's container type, so their lowering is
# the wrapper `MacroTracer._transform_ast_lambda_for_macro` builds. Spelled out once.
_FUNCTOR = (
    "(lambda __ps_c: (type(__ps_c) if type(__ps_c) in (frozenset, list, set, tuple) "
    "else (lambda __ps_i: __ps_i))({builtin}((lambda __ps_e: {body}), __ps_c)))({seed})"
)


@pytest.mark.parametrize("code,expected", MACRO_LOWERINGS)
def test_lower_macro_stages(code: str, expected: str) -> None:
    assert lower_pipelines(code) == expected


@pytest.mark.parametrize(
    "code,builtin,body,seed",
    [
        ("[1] |> map[$ * 2] |> $.", "map", "__ps_e * 2", "([1])"),
        ("[1] |> filter[$ > 0] |> $.", "filter", "__ps_e > 0", "([1])"),
    ],
)
def test_lower_eager_functor_macros(code, builtin, body, seed) -> None:
    inner = _FUNCTOR.format(builtin=builtin, body=body, seed=seed)
    assert lower_pipelines(code) == f"({inner})."


# Compose and partial application denote functions, so each lowers to that lambda.
COMPOSE_LOWERINGS = [
    ("h = f .> g\n", "h = (lambda *__ps_a, **__ps_k: (g)(((f))(*__ps_a, **__ps_k)))\n"),
    ("h = f <. g\n", "h = (lambda *__ps_a, **__ps_k: ((f))((g)(*__ps_a, **__ps_k)))\n"),
    (
        "h = 5 $> pow\n",
        "h = (lambda *__ps_a, **__ps_k: (pow)((5), *__ps_a, **__ps_k))\n",
    ),
    (
        "h = pow <$ 5\n",
        "h = (lambda *__ps_a, **__ps_k: ((pow))(5, *__ps_a, **__ps_k))\n",
    ),
]


@pytest.mark.parametrize("code,expected", COMPOSE_LOWERINGS)
def test_lower_compose_and_partial_ops(code: str, expected: str) -> None:
    assert lower_pipelines(code) == expected


BAILS = [
    # nothing to lower
    "no pipeline here.",
    "xs |> sorted |> lis",  # final stage is an ordinary expression
    'x = "a |> b"\nx.',  # the pipe lives inside a string
    # the chain denotes a function, not a value
    "$ |> sorted |> $.",
    "|> sorted |> $.",
    "xs |> f$(1) |> $.",
    "xs |>> na",
    # function exponentiation has no finite expression to hand a static analyzer
    "xs |> collatz .** 20 |> $.",
    # the cursor is on the placeholder itself; substituting would rewrite the
    # very word being completed
    '["a"] |> $',
    '["a"] |> f($na',
    # the cursor is inside a macro's own scope, completing against *its* lambda
    "xs |> map[$.",
    # a macro whose effect on the piped value we don't model
    "xs |> reduce[$ + $] |> $.",
    "xs |> fork[$ |> a, $ |> b] |> $.",
    "xs |> future[$ * 2] |> $.",
    # a macro body that is itself a pipeline: its `$` is the macro's, not ours
    "xs |> map[$ |> f] |> $.",
    # distinct placeholders make the stage (or a macro body) a multi-arg function
    '["a"] |> f($v, $w) |> $.',
    '["a"] |> f($, $) |> $.',
    "xs |> map[$ + $] |> $.",
    # a placeholder stage under a non-plain pipe is a multi-arg / reversed apply
    "xs *|> $.foo",
    # empty stages
    "xs |> f($) |> ",
]


@pytest.mark.parametrize("code", BAILS)
def test_lower_pipelines_bails(code: str) -> None:
    assert lower_pipelines(code) is None


# `(pipescript source, plain-Python expected value)`. The lowering models each
# macro's effect on the piped value, so the two must agree -- including the
# container type the eager `map`/`filter` restore, and the laziness of `imap`.
RUNTIME_AGREEMENT = [
    "[1, 2, 3] |> map[$ * 2]",
    "{1, 2} |> map[$ * 2]",
    "(1, 2) |> map[$ * 2]",
    "[1, 2, 3] |> filter[$ > 1]",
    "{1, 2, 3} |> filter[$ > 1]",
    "[1, 2] |> imap[$ * 2] |> list",
    "[1, 2] |> ifilter[$ > 1] |> list",
    "[1, 2] |> do[len($)]",
    "[1, 2] |> when[len($) > 1]",
    "[1, 2] |> f[$ + [3]]",
    "(len .> str) <| [1, 2]",
    "(str <. len) <| [1, 2]",
    "(2 $> pow) <| 5",
    "(pow <$ 2) <| 5",
]


@pytest.mark.parametrize("code", RUNTIME_AGREEMENT)
def test_macro_lowerings_match_the_runtime(code: str) -> None:
    """The lowering is only useful if it denotes what the pipeline actually does.
    Run both and compare: an inaccurate lowering hands jedi a wrong *type*, which is
    worse than the bail-out it replaced."""
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    assert shell.run_cell("%load_ext pipescript").error_in_exec is None
    result = shell.run_cell(code, store_history=True)
    assert result.error_in_exec is None, result.error_in_exec

    # a trailing newline makes the chain complete, so it lowers in value mode
    lowered = lower_pipelines(code + "\n")
    assert lowered is not None, code
    actual = eval(compile(lowered.strip(), "<lowered>", "eval"), {})

    expected = result.result
    assert type(actual) is type(expected), (code, lowered)
    if isinstance(expected, (list, set, tuple, str, int)):
        assert actual == expected, (code, lowered)
    else:  # a lazy iterator compares by exhaustion, not identity
        assert list(actual) == list(expected), (code, lowered)


def test_completion_token_and_tail_guard() -> None:
    assert cp._completion_token("xs |> $.up") == (".", "up")
    assert cp._completion_token("xs |> $.") == (".", "")
    assert cp._completion_token("xs |> $") == ("$", "")
    # substituting the placeholder would replace the token under the cursor
    assert cp._tail_word_preserved("xs |> $.up", "(xs).up")
    assert not cp._tail_word_preserved("xs |> $", "(xs)")


def test_completion_sources_prefers_lowered_then_legacy() -> None:
    tracers = _tracers()
    assert cp._completion_sources("[1, 2] |> sum |> $.bit_", tracers) == [
        "((sum)(([1, 2]))).bit_",
        "[1, 2] | sum | _.bit_",
    ]
    # nothing lowerable: only the legacy transform, and vanilla Python is untouched
    assert cp._completion_sources("xs |> reduce[$+$] |> $.x", tracers) == [
        "xs | reduce[_+_] | _.x"
    ]
    assert cp._completion_sources("os.pa", tracers) == ["os.pa"]


class _FakeCompletion:
    def __init__(self, start: int, end: int) -> None:
        self.start, self.end = start, end


class _EmptyThenMatchCompleter:
    """First source yields nothing, second yields a match -- the fallback path."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def completions(self, text: str, offset: int) -> list[Any]:
        self.seen.append(text)
        if len(self.seen) == 1:
            return []
        return [_FakeCompletion(offset - 4, offset)]


def test_empty_lowered_result_falls_back_to_legacy() -> None:
    shell = ShellNoKernel()
    shell.Completer = _EmptyThenMatchCompleter()  # type: ignore[assignment]
    patch_completer(shell, _tracers())
    try:
        text = "[1, 2] |> sum |> $.bit_"
        (completion,) = shell.Completer.completions(text, len(text))
    finally:
        unpatch_completer(shell)
    lowered, legacy = cp._completion_sources(text, _tracers())
    assert shell.Completer.seen == [lowered, legacy]  # type: ignore[attr-defined]
    # offsets are shifted by the length of whichever source actually answered
    assert (completion.start, completion.end) == (len(text) - 4, len(text))


def _fresh_patched_shell():
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    patch_completer(shell, _tracers())
    return shell


def _complete(shell, text: str) -> tuple[list[str], tuple[int, int]]:
    from IPython.core.completer import provisionalcompleter

    with provisionalcompleter():
        completions = list(shell.Completer.completions(text, len(text)))
    first = completions[0]
    return [c.text for c in completions], (
        first.start - len(text),
        first.end - len(text),
    )


@pytest.mark.parametrize(
    "code,expected",
    [
        ('["a", "b", "c"] |> "\\n".join($) |> $.up', "upper"),
        ("[1, 2] |> sum |> $.bit_", "bit_length"),
        ('{"a": 1} **|> dict |> $.ke', "keys"),
        ('d = {"a": 1} |> $.items() |> list |> $[0].c', "count"),
        # a name bound by a pipeline on an earlier line
        ('result = ["a", "b"] |> "\\n".join($)\nresult.up', "upper"),
        ("xs = [3, 1] |> sorted\nys = xs |> len\nys.bit_", "bit_length"),
        # ...including one bound by `|>>`
        ("[1, 2] |>> nums |> sum\nnums.co", "count"),
    ],
)
def test_completes_without_running_the_prefix(code: str, expected: str) -> None:
    """The whole point: jedi types the pipeline statically, so the completion no
    longer depends on `_` holding the result of a previously-run prefix."""
    pytest.importorskip("jedi")
    shell = _fresh_patched_shell()
    try:
        shell.user_ns["_"] = 42  # poison the runtime fallback
        matches, (start, end) = _complete(shell, code)
        assert expected in matches
        # the offsets the completer hands back must span exactly the typed word
        _, word = cp._completion_token(code)
        assert (start, end) == (-len(word), 0)
    finally:
        unpatch_completer(shell)


def test_completion_does_not_register_brace_block_bodies() -> None:
    """Completion is analysis, not execution: it must pass ``pure=True`` so a
    ``macro{...}`` body isn't stashed in the tracer's process-global registry."""
    shell = _fresh_patched_shell()
    before = dict(BraceBlockTracer.block_sources), BraceBlockTracer._counter
    try:
        from IPython.core.completer import provisionalcompleter

        text = "xs |> do{ print($) } |> $.up"
        with provisionalcompleter():
            list(shell.Completer.completions(text, len(text)))
    finally:
        unpatch_completer(shell)
    assert (dict(BraceBlockTracer.block_sources), BraceBlockTracer._counter) == before
