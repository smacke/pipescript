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
from bisect import bisect_left
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


def _by_first_char(tokens: list[str]) -> dict[str, list[str]]:
    """Index longest-match candidates by their first character, so the scanner tries
    a handful of prefixes per character instead of the whole table."""
    index: dict[str, list[str]] = {}
    for token in tokens:
        index.setdefault(token[0], []).append(token)
    return index


_OPS_BY_FIRST = _by_first_char(_ALL_OPS)
_PUNCT_BY_FIRST = _by_first_char(_PUNCT)

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


class _Scan(NamedTuple):
    toks: list[_Tok]
    # bracket stack left open at the end of the source, i.e. the stack at the cursor
    cursor_stack: tuple[int, ...]
    # the source with string and comment characters blanked out
    blanked: str
    # offset of each bracket's closer, keyed by the offset of its opener; an
    # unclosed bracket is absent, so its region runs to the end of the source
    close_of: dict[int, int]
    # sorted starts of the tokens that make a span unlowerable (`op` and `bail`),
    # and of the placeholders -- both queried by range, so both want a bisect
    barrier_starts: list[int]
    ph_starts: list[int]
    phs: list[_Tok]


def _scan(code: str) -> _Scan | None:
    """Tokens of interest in ``code``. ``None`` if the source cannot be tokenized."""
    blanked = _blank_strings(code)
    if blanked is None:
        return None
    code = blanked
    toks: list[_Tok] = []
    stack: list[int] = []
    close_of: dict[int, int] = {}
    i, n = 0, len(code)
    while i < n:
        char = code[i]
        if char in _OPENERS:
            stack.append(i)
            i += 1
            continue
        if char in _CLOSERS:
            if stack:
                close_of[stack.pop()] = i
            i += 1
            continue
        if char == "\n":
            # Inside brackets a newline is an implicit continuation; at depth zero
            # (and absent a backslash) it ends the logical line, hence the chain.
            if not stack and not (i > 0 and code[i - 1] == "\\"):
                toks.append(_Tok("stop", i, i + 1, (), "\n"))
            i += 1
            continue
        matched = False
        for op in _OPS_BY_FIRST.get(char, ()):
            if code.startswith(op, i):
                kind = "op" if op in _PIPE_OPS else "bail"
                toks.append(_Tok(kind, i, i + len(op), tuple(stack), op))
                i += len(op)
                matched = True
                break
        if matched:
            continue
        if char == "$":
            # ``$$`` is a macro-template placeholder, not a pipeline one.
            if code.startswith("$$", i):
                i += 2
                continue
            # ``f$(x)`` is a partial call -- a curried *function*, so a stage
            # containing one isn't ours to lower. (Detached, as in ``f $(x)``, it
            # really is a placeholder; that's pyccolo's own rule, see
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
        if char.isalpha() or char == "_":
            match = _NAME_RE.match(code, i)
            assert match is not None
            if match.group() in _stop_keywords():
                toks.append(_Tok("stop", i, match.end(), tuple(stack), match.group()))
            i = match.end()
            continue
        if char.isdigit():
            match = _NUM_RE.match(code, i)
            assert match is not None
            i = match.end()
            continue
        for punct in _PUNCT_BY_FIRST.get(char, ()):
            if code.startswith(punct, i):
                if punct in _STOP_OPS:
                    toks.append(_Tok("stop", i, i + len(punct), tuple(stack), punct))
                i += len(punct)
                matched = True
                break
        if not matched:
            i += 1
    barrier_starts = [t.start for t in toks if t.kind in ("op", "bail")]
    phs = [t for t in toks if t.kind == "ph"]
    return _Scan(
        toks,
        tuple(stack),
        blanked,
        close_of,
        barrier_starts,
        [p.start for p in phs],
        phs,
    )


def _any_in(starts: list[int], lo: int, hi: int) -> bool:
    return bisect_left(starts, lo) < bisect_left(starts, hi)


def _is_opaque(scan: _Scan, lo: int, hi: int, macros: frozenset[str]) -> bool:
    """True if ``[lo, hi)`` holds a construct whose placeholders are not ours to
    substitute: a nested chain, a compose/partial operator, or a macro scope."""
    if _any_in(scan.barrier_starts, lo, hi):
        return True
    text = scan.blanked[lo:hi]
    if _BRACE_BLOCK_RE.search(text):
        return True
    for match in _NAME_RE.finditer(text):
        if match.group() in macros and text[match.end() :].lstrip().startswith("["):
            return True
    return False


def _placeholders(scan: _Scan, lo: int, hi: int) -> list[_Tok]:
    return scan.phs[bisect_left(scan.ph_starts, lo) : bisect_left(scan.ph_starts, hi)]


def _substitute(code: str, lo: int, hi: int, phs: list[_Tok], expr: str) -> str:
    out, prev = [], lo
    for ph in phs:
        out.append(code[prev : ph.start])
        out.append(expr)
        prev = ph.end
    out.append(code[prev:hi])
    return "".join(out)


class _Chain(NamedTuple):
    """A maximal run of pipe operators at one bracket depth, bounded by the stop
    tokens (or the enclosing bracket) around it. ``start``/``end`` delimit the whole
    chain expression, seed included."""

    start: int
    end: int
    ops: list[_Tok]

    @property
    def depth(self) -> int:
        return len(self.ops[0].stack)


def _chains(scan: _Scan, n: int) -> list[_Chain]:
    """Every pipeline chain in the source, innermost-and-rightmost first.

    Two ops belong to the same chain iff they sit at the same bracket depth with no
    stop token between them: ``|`` binds tighter than ``==``, so ``a == b |> f`` is
    two expressions, only one of which is a chain.
    """
    by_stack: dict[tuple[int, ...], list[_Tok]] = {}
    for tok in scan.toks:
        if tok.kind == "op":
            by_stack.setdefault(tok.stack, []).append(tok)

    stops_by_stack: dict[tuple[int, ...], list[_Tok]] = {}
    for tok in scan.toks:
        if tok.kind == "stop" and tok.stack in by_stack:
            stops_by_stack.setdefault(tok.stack, []).append(tok)

    chains: list[_Chain] = []
    for stack, ops in by_stack.items():
        region_start = stack[-1] + 1 if stack else 0
        region_end = scan.close_of.get(stack[-1], n) if stack else n
        stops = [
            tok
            for tok in stops_by_stack.get(stack, ())
            if region_start <= tok.start < region_end
        ]
        starts = [s.start for s in stops]
        group: list[_Tok] = []
        for op in ops:
            if group and _any_in(starts, group[-1].end, op.start):
                chains.append(
                    _chain_span(group, stops, starts, region_start, region_end)
                )
                group = []
            group.append(op)
        if group:
            chains.append(_chain_span(group, stops, starts, region_start, region_end))
    chains.sort(key=lambda c: (c.depth, c.start), reverse=True)
    return chains


def _chain_span(
    ops: list[_Tok],
    stops: list[_Tok],
    starts: list[int],
    region_start: int,
    region_end: int,
) -> _Chain:
    # the chain runs from just after the stop preceding its seed to just before the
    # stop following its final stage
    before = bisect_left(starts, ops[0].start)
    start = stops[before - 1].end if before else region_start
    after = bisect_left(starts, ops[-1].end)
    end = stops[after].start if after < len(stops) else region_end
    return _Chain(max(start, region_start), min(end, region_end), ops)


def _fold(
    code: str, scan: _Scan, chain: _Chain, macros: frozenset[str], cursor_mode: bool
) -> str | None:
    """The nested-call expression a chain denotes, or ``None`` if it holds anything
    we cannot lower soundly. Under ``cursor_mode`` the final stage is the partially
    typed text under the cursor, which constrains what may be rewritten there."""
    ops = chain.ops
    # ``[chain.start, ops[0].start)`` is the seed; ``[ops[i].end, ops[i+1].start)``
    # is the stage that op ``i`` applies. The final stage runs to ``chain.end``.
    stage_bounds = [
        (op.end, ops[i + 1].start if i + 1 < len(ops) else chain.end)
        for i, op in enumerate(ops)
    ]
    seed = code[chain.start : ops[0].start]

    if (
        not seed.strip()
        or _is_opaque(scan, chain.start, ops[0].start, macros)
        or _placeholders(scan, chain.start, ops[0].start)
    ):
        # An empty seed is a leading-``|>`` thunk and a seed with a placeholder is
        # an undetermined pipeline: both denote a function, not a value.
        return None

    expr = f"({seed.strip()})"
    for i, op in enumerate(ops):
        kind = _PIPE_OPS[op.text]
        lo, hi = stage_bounds[i]
        stage = code[lo:hi].strip()
        last = i == len(ops) - 1

        if kind == "pipe_assign":
            # ``|>>`` binds a name and yields its LHS untouched, which is exactly
            # what a walrus does -- and it lets the bound name be typed downstream.
            if (cursor_mode and last) or not stage.isidentifier():
                return None
            new = f"({stage} := {expr})"
            if last:
                return new
            expr = new
            continue
        if _is_opaque(scan, lo, hi, macros):
            return None
        phs = _placeholders(scan, lo, hi)

        if cursor_mode and last:
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
            # In cursor mode the final stage carries the cursor, so only its leading
            # whitespace may be touched -- `.strip()` would eat a trailing partial
            # token's context.
            new = new.lstrip() if (cursor_mode and last) else new.strip()
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
            return new
        expr = f"({new})"
    return None


def _splice(code: str, chain: _Chain, folded: str) -> str:
    seed = code[chain.start : chain.ops[0].start]
    # The chain can begin a suite-indented statement, so carry its indentation over
    # rather than dedenting the rewritten line into a syntax error.
    indent = seed[: len(seed) - len(seed.lstrip())]
    return code[: chain.start] + indent + folded + code[chain.end :]


# Each round strips a whole nesting level, so this bounds pipeline nesting depth
# rather than chain count -- a backstop against a fold that fails to make progress.
_MAX_ROUNDS = 32


def _lower_complete_chains(code: str, tail_is_cursor: bool) -> str | None:
    """Lower every complete chain that can be folded, in one pass, or ``None``.

    A foldable chain never contains another chain -- a stage or seed holding a pipe
    operator is opaque, so its chain would not fold. The chains picked here are
    therefore pairwise disjoint and can all be spliced in a single right-to-left
    sweep, which is what keeps this linear in the size of the cell rather than
    quadratic in the number of chains. Nesting still costs a round: an outer chain
    only becomes foldable once the chain inside it has become plain Python.
    """
    scan = _scan(code)
    if scan is None:
        return None
    macros = _macro_names()
    n = len(code)
    picks: list[tuple[_Chain, str]] = []
    for chain in _chains(scan, n):
        # When the source ends at a cursor, a chain running to that end is (or
        # encloses) the one being typed into, so its last stage is incomplete.
        if tail_is_cursor and chain.end >= n:
            continue
        folded = _fold(code, scan, chain, macros, cursor_mode=False)
        if folded is not None:
            picks.append((chain, folded))
    if not picks:
        return None
    for chain, folded in sorted(picks, key=lambda pick: pick[0].start, reverse=True):
        code = _splice(code, chain, folded)
    return code


def _lower_cursor_chain(code: str) -> str | None:
    """Lower the chain the cursor sits in: the innermost one running to the end of
    the source. Its final stage is partially typed, so it folds under the stricter
    cursor-mode rules."""
    scan = _scan(code)
    if scan is None:
        return None
    n = len(code)
    for chain in _chains(scan, n):
        if chain.end < n:
            continue
        # ``_chains`` is innermost-first, so the first chain reaching the cursor is
        # the one it belongs to. An enclosing chain also reaches the end, but its
        # final stage is the one we just tried to fold.
        folded = _fold(code, scan, chain, _macro_names(), cursor_mode=True)
        return None if folded is None else _splice(code, chain, folded)
    return None


def lower_pipelines(code: str, cursor_at_end: bool = True) -> str | None:
    """Rewrite ``code``'s pipeline chains into the equivalent nested-call Python, so
    a static analyzer can type what flows through them. ``None`` means nothing was
    lowerable -- the caller should fall back to the plain lexical transform.

    Complete chains lower to values, which is what lets a name bound by a pipeline
    on one line be typed on the next. Chains are lowered innermost-first, so a
    nested chain becomes plain Python before the chain enclosing it is folded.

    With ``cursor_at_end`` (the default) ``code`` is the source left of the cursor,
    and the chain it ends inside gets a final, stricter pass that leaves the token
    being completed untouched. Pass ``False`` for source that holds no cursor.
    """
    if not any(op in code for op in ("|>", "?>", "<|", "<?")):
        return None
    original = code
    for _ in range(_MAX_ROUNDS):
        lowered = _lower_complete_chains(code, tail_is_cursor=cursor_at_end)
        if lowered is None or lowered == code:
            break
        code = lowered
    if cursor_at_end:
        lowered = _lower_cursor_chain(code)
        if lowered is not None:
            code = lowered
    return code if code != original else None
