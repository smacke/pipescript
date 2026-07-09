"""Rewrite a pipeline chain into the equivalent nested-call Python, so that a
static analyzer (jedi, via IPython's completer) can type the value flowing
through it without the pipeline ever having been run.

``pyc.transform`` lowers ``|>`` to ``|`` and ``$`` to ``_``, which is valid but
meaningless Python: ``["a"] | "\\n".join(_)`` is a type error, so jedi infers
nothing and attribute completion on a later stage falls back to whatever the
IPython ``_`` variable happens to hold. Pipes are just function application, so
each chain has an exact, side-effect-free Python equivalent::

    ["a", "b"] |> "\\n".join($) |> $.   ->   ( "\\n".join((["a", "b"])) ).

The rewrite substitutes the piped value in at each placeholder, which never
disturbs the trailing identifier the completer is completing -- see
``_tail_word_preserved`` in ``pipescript.patches.completion_patch``, which makes
that a checked precondition rather than a happy accident.

The input is the source *left of the cursor*, so it is routinely unparseable
(trailing dot, unbalanced parens). jedi's parser recovers from that; ``ast.parse``
does not. So everything here works on tokens and character offsets, and bails out
(returning ``None``, leaving the caller with today's behavior) on any construct it
cannot lower soundly.
"""

from __future__ import annotations

import re
import tokenize
from functools import lru_cache
from typing import NamedTuple

from pyccolo.syntax_augmentation import _line_starts, make_tokens_by_line, offset_of

__all__ = ["lower_pipelines"]


class _Tok(NamedTuple):
    # "op" (a lowerable pipe), "bail" (an operator we decline to lower), "ph" (a
    # ``$`` placeholder), or "stop" (a token that cannot appear inside a pipeline
    # chain, so it bounds one).
    kind: str
    start: int
    end: int
    # offsets of the brackets enclosing this token; its depth is the length
    stack: tuple[int, ...]
    text: str


# Pipe operators we can lower, mapped to how the stage consumes the piped value.
# ``?>`` short-circuits on ``None`` at runtime, which doesn't change the type we
# want to complete against, so it lowers like ``|>``.
_PIPE_OPS = {
    "**|>": "pipe_dict",
    "*|>": "pipe_tuple",
    "|>>": "pipe_assign",
    "|>": "pipe",
    "**?>": "pipe_dict",
    "*?>": "pipe_tuple",
    "?>": "pipe",
    "<|**": "apply_dict",
    "<|*": "apply_tuple",
    "<|": "apply",
    "<?**": "apply_dict",
    "<?*": "apply_tuple",
    "<?": "apply",
}

# Compose and partial-apply operators evaluate to *functions* rather than values;
# lowering them soundly means synthesizing lambdas. Recognized only so we can
# refuse to lower a chain containing one.
_BAIL_OPS = frozenset(
    {
        "**.>",
        "*.>",
        ".>",
        "<.**",
        "<.*",
        "<.",
        "**$>",
        "*$>",
        "$>",
        "<$**",
        "<$*",
        "<$",
        ".**",
    }
)

_ALL_OPS = sorted(set(_PIPE_OPS) | _BAIL_OPS, key=len, reverse=True)

_OPENERS = "([{"
_CLOSERS = ")]}"

# Bind tighter than ``|``, so they sit *inside* a pipeline stage. Listed only so
# that longest-match consumes them before the single-character stops below (``>>``
# must not be read as ``>``).
_SKIP_OPS = (">>", "<<", "**", "//")

# Bind looser than ``|`` (or are statement punctuation), so a chain cannot span
# one: ``a == b |> f`` parses as ``a == (b |> f)``.
_STOP_OPS = (
    ",",
    ";",
    ":=",
    ":",
    "==",
    "!=",
    "<=",
    ">=",
    "<",
    ">",
    "=",
    "->",
    "+=",
    "-=",
    "*=",
    "/=",
    "//=",
    "**=",
    "%=",
    "@=",
    "&=",
    "|=",
    "^=",
    ">>=",
    "<<=",
)

_PUNCT = sorted(set(_SKIP_OPS) | set(_STOP_OPS), key=len, reverse=True)

_NAME_RE = re.compile(r"[A-Za-z_]\w*")
_NUM_RE = re.compile(r"\d[\w.]*")
# A macro's subscript (``map[$ * 2]``) or brace block (``do{ ... }``) is its own
# scope: the ``$`` inside binds to the macro's lambda, not to the piped value.
_BRACE_BLOCK_RE = re.compile(r"[A-Za-z_]\w*\s*\{")


_EXTRA_STOP_KEYWORDS = frozenset({"for", "as", "with", "import", "from", "async"})


@lru_cache(maxsize=1)
def _stop_keywords() -> frozenset[str]:
    # Imported lazily: ``pipeline_tracer`` imports from ``pipescript.analysis``.
    from pipescript.tracers.pipeline_tracer import PipelineTracer

    return frozenset(PipelineTracer._LEADING_STARTER_KEYWORDS) | _EXTRA_STOP_KEYWORDS


def _macro_names() -> frozenset[str]:
    from pipescript.tracers.macro_tracer import MacroTracer

    return frozenset(
        set(MacroTracer.static_macros)
        | set(MacroTracer.dynamic_macros)
        | set(MacroTracer.dynamic_method_macros)
    )


def _blank_strings(code: str) -> str | None:
    """``code`` with every string and comment character replaced by a space, so
    neither the scanner nor the regexes below mistake their contents for syntax.
    ``make_tokens_by_line`` is IPython's tolerant tokenizer -- it survives the
    unbalanced brackets and trailing dots a cursor prefix routinely ends in."""
    chars = list(code)
    try:
        starts = _line_starts(code)
        n_lines = len(starts)
        for line in make_tokens_by_line(code.splitlines(keepends=True)):
            for tok in line:
                name = tokenize.tok_name.get(tok.type, "")
                if not (
                    tok.type in (tokenize.STRING, tokenize.COMMENT)
                    or name.startswith("FSTRING")
                ):
                    continue
                if tok.start[0] > n_lines or tok.end[0] > n_lines:
                    continue
                lo = offset_of(starts, *tok.start)
                hi = offset_of(starts, *tok.end)
                for k in range(max(lo, 0), min(hi, len(code))):
                    if chars[k] != "\n":
                        chars[k] = " "
    except Exception:
        return None
    return "".join(chars)


def _scan(code: str) -> tuple[list[_Tok], tuple[int, ...], str] | None:
    """Tokens of interest in ``code``, the bracket stack left open at its end
    (i.e. the stack at the cursor), and the string-blanked source. ``None`` if the
    source cannot be tokenized."""
    blanked = _blank_strings(code)
    if blanked is None:
        return None
    code = blanked
    toks: list[_Tok] = []
    stack: list[int] = []
    i, n = 0, len(code)
    while i < n:
        char = code[i]
        if char in _OPENERS:
            stack.append(i)
            i += 1
            continue
        if char in _CLOSERS:
            if stack:
                stack.pop()
            i += 1
            continue
        if char == "\n":
            # Inside brackets a newline is an implicit continuation; at depth zero
            # (and absent a backslash) it ends the logical line, hence the chain.
            if not stack and not (i > 0 and code[i - 1] == "\\"):
                toks.append(_Tok("stop", i, i + 1, (), "\n"))
            i += 1
            continue
        for op in _ALL_OPS:
            if code.startswith(op, i):
                kind = "op" if op in _PIPE_OPS else "bail"
                toks.append(_Tok(kind, i, i + len(op), tuple(stack), op))
                i += len(op)
                break
        else:
            if char == "$":
                # ``$$`` is a macro-template placeholder, not a pipeline one.
                if code.startswith("$$", i):
                    i += 2
                    continue
                # ``f$(x)`` is a partial call -- a curried *function*, so a stage
                # containing one isn't ours to lower. (Detached, as in ``f $(x)``,
                # it really is a placeholder; that's pyccolo's own rule, see
                # ``PipelineTracer._non_partial_call_op_spec``.)
                if code.startswith("$(", i) and i > 0 and not code[i - 1].isspace():
                    toks.append(_Tok("bail", i, i + 2, tuple(stack), "$("))
                    i += 2
                    continue
                match = _NAME_RE.match(code, i + 1)
                end = match.end() if match else i + 1
                toks.append(_Tok("ph", i, end, tuple(stack), code[i:end]))
                i = end
                continue
            match = _NAME_RE.match(code, i)
            if match:
                if match.group() in _stop_keywords():
                    toks.append(
                        _Tok("stop", i, match.end(), tuple(stack), match.group())
                    )
                i = match.end()
                continue
            match = _NUM_RE.match(code, i)
            if match:
                i = match.end()
                continue
            for punct in _PUNCT:
                if code.startswith(punct, i):
                    if punct in _STOP_OPS:
                        toks.append(
                            _Tok("stop", i, i + len(punct), tuple(stack), punct)
                        )
                    i += len(punct)
                    break
            else:
                i += 1
    return toks, tuple(stack), blanked


def _is_opaque(toks: list[_Tok], blanked: str, lo: int, hi: int) -> bool:
    """True if ``blanked[lo:hi]`` holds a construct whose placeholders are not ours
    to substitute: a nested chain, a compose/partial operator, or a macro scope."""
    if any(tok.kind in ("op", "bail") and lo <= tok.start < hi for tok in toks):
        return True
    text = blanked[lo:hi]
    if _BRACE_BLOCK_RE.search(text):
        return True
    names = _macro_names()
    for match in _NAME_RE.finditer(text):
        if match.group() in names and text[match.end() :].lstrip().startswith("["):
            return True
    return False


def _placeholders(toks: list[_Tok], lo: int, hi: int) -> list[_Tok]:
    return [tok for tok in toks if tok.kind == "ph" and lo <= tok.start < hi]


def _substitute(code: str, lo: int, hi: int, phs: list[_Tok], expr: str) -> str:
    out, prev = [], lo
    for ph in phs:
        out.append(code[prev : ph.start])
        out.append(expr)
        prev = ph.end
    out.append(code[prev:hi])
    return "".join(out)


def _locate_chain(
    toks: list[_Tok], cursor_stack: tuple[int, ...]
) -> tuple[int, list[_Tok]] | None:
    """Find the pipeline chain the cursor sits in, walking outward from its
    innermost enclosing bracket. Returns ``(chain_start, ops)``.

    Walking outward is what lets the cursor be *deeper* than the chain, as in
    ``foo |> bar($, baz.`` -- the ops are at depth 0, the cursor inside ``bar(``.
    """
    for depth in range(len(cursor_stack), -1, -1):
        stack = cursor_stack[:depth]
        region_start = cursor_stack[depth - 1] + 1 if depth else 0
        # Any stop token before the cursor breaks the chain, so the chain begins
        # after the last one.
        chain_start = max(
            [region_start]
            + [
                tok.end
                for tok in toks
                if tok.kind == "stop"
                and tok.stack == stack
                and tok.start >= region_start
            ]
        )
        ops = [
            tok
            for tok in toks
            if tok.kind == "op" and tok.stack == stack and tok.start >= chain_start
        ]
        if ops:
            return chain_start, ops
    return None


def lower_pipelines(code: str) -> str | None:
    """Rewrite the pipeline chain under the cursor (i.e. at the end of ``code``)
    into equivalent nested-call Python. ``None`` means "no lowerable chain here" --
    the caller should fall back to the plain lexical transform.
    """
    if not any(op in code for op in ("|>", "?>", "<|", "<?")):
        return None
    scanned = _scan(code)
    if scanned is None:
        return None
    toks, cursor_stack, blanked = scanned
    located = _locate_chain(toks, cursor_stack)
    if located is None:
        return None
    chain_start, ops = located

    # ``[chain_start, ops[0].start)`` is the seed; ``[ops[i].end, ops[i+1].start)``
    # is the stage that op ``i`` applies. The final stage runs to the cursor.
    stage_bounds = [
        (op.end, ops[i + 1].start if i + 1 < len(ops) else len(code))
        for i, op in enumerate(ops)
    ]
    seed = code[chain_start : ops[0].start]

    if (
        not seed.strip()
        or _is_opaque(toks, blanked, chain_start, ops[0].start)
        or _placeholders(toks, chain_start, ops[0].start)
    ):
        # An empty seed is a leading-``|>`` thunk and a seed with a placeholder is
        # an undetermined pipeline: both denote a function, not a value.
        return None

    # The chain can begin a suite-indented statement, so carry its indentation over
    # rather than dedenting the rewritten line into a syntax error.
    indent = seed[: len(seed) - len(seed.lstrip())]
    expr = f"({seed.strip()})"
    for i, op in enumerate(ops):
        kind = _PIPE_OPS[op.text]
        lo, hi = stage_bounds[i]
        stage = code[lo:hi].strip()
        last = i == len(ops) - 1

        if kind == "pipe_assign":
            # ``|>>`` binds a name and yields its LHS untouched.
            if last:
                return None
            continue
        if _is_opaque(toks, blanked, lo, hi):
            return None
        phs = _placeholders(toks, lo, hi)

        if last:
            # A final stage with no placeholder is an ordinary expression in the
            # enclosing scope; the plain transform already completes it correctly,
            # and lowering would only rewrite the token being completed.
            if not phs:
                return None
            # The cursor is *on* the placeholder (``xs |> $`` / ``xs |> f($na``);
            # substituting would replace the very word being completed.
            if phs[-1].end == hi:
                return None

        if phs:
            if kind != "pipe":
                # a placeholder stage under ``*|>``/``**|>``/``<|`` is a multi-arg
                # or reversed application; not worth guessing at
                return None
            names = {ph.text for ph in phs}
            if len(names) > 1 or (names == {"$"} and len(phs) > 1):
                # distinct placeholders make the stage a multi-arg function
                return None
            new = _substitute(code, lo, hi, phs, expr)
            if not last:
                new = new.strip()
        elif not stage:
            return None
        elif kind.startswith("apply"):
            star = {"apply": "", "apply_tuple": "*", "apply_dict": "**"}[kind]
            # ``f <| x``: the accumulated expr is the function, the stage the value
            new = f"({expr})({star}{stage})"
        else:
            star = {"pipe": "", "pipe_tuple": "*", "pipe_dict": "**"}[kind]
            new = f"({stage})({star}{expr})"

        if last:
            return code[:chain_start] + indent + new.lstrip()
        expr = f"({new})"
    return None
