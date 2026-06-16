"""Pipescript-aware enrichment of runtime errors.

When a pipeline stage fails, the bare Python exception says nothing about *which*
stage failed or that the offending object is a pipescript stage-function. The
helpers here attach short, pipescript-level notes to such exceptions, used from
two places:

* the runtime apply-site wrapper in :class:`~pipescript.tracers.pipeline_tracer.
  PipelineTracer` (precise: it has the pipe ``BinOp`` node and the piped value);
* the display-time annotator invoked from the IPython ``showtraceback`` patch in
  :mod:`pipescript.extension` (a backstop that also covers errors surfacing far
  from the apply site).

A canonical footgun: ``fork``/``parallel`` branches must be *functions*, so a
branch composes (``map[pred] .> all``). Writing ``map[pred] |> all`` instead
*applies* ``all`` to the ``map[pred]`` stage-function, i.e. ``all(<function>)``
-> ``TypeError: 'function' object is not iterable``. The hint names this.
"""

from __future__ import annotations

import ast
import re
from types import TracebackType
from typing import Any

#: Macros whose subscript template elements must evaluate to functions.
_FORK_MACROS = frozenset({"fork", "parallel"})

#: ``'<type>' object is not iterable|subscriptable|callable`` -- the shapes that
#: show up when a function/partial lands where a value was expected.
_NOT_A_VALUE_RE = re.compile(
    r"'(?P<typ>[\w.]+)' object is not (?:iterable|subscriptable)"
)

#: Type names (from the message above) that indicate the stray object is itself
#: callable -- i.e. a stage/partial leaked into a value position.
_CALLABLE_TYPE_NAMES = frozenset(
    {
        "function",
        "builtin_function_or_method",
        "method",
        "partial",
        "cython_function_or_method",
        "method-wrapper",
    }
)


def add_note(exc: BaseException, note: str) -> None:
    """Attach a pipescript note to ``exc`` (idempotent).

    Notes are recorded on ``exc._pyc_notes`` (ordered, de-duplicated) so callers
    and tests can read them on any Python version, and also forwarded to the
    native :meth:`BaseException.add_note` (3.11+) so non-IPython hosts render
    them automatically.
    """
    notes = getattr(exc, "_pyc_notes", None)
    if notes is None:
        notes = []
        try:
            exc._pyc_notes = notes  # type: ignore[attr-defined]
        except Exception:
            return
    if note in notes:
        return
    notes.append(note)
    native_add = getattr(exc, "add_note", None)
    if callable(native_add):
        try:
            native_add(note)
        except Exception:
            pass


def _operator_token(node: ast.AST | None) -> str | None:
    """The surface operator (``|>``, ``.>``, ...) for a lowered pipe ``BinOp``."""
    if node is None:
        return None
    try:
        import pyccolo as pyc

        augs = pyc.BaseTracer.get_augmentations(id(node))
        for aug in augs:
            if getattr(aug, "aug_type", None) == pyc.AugmentationType.binop:
                return aug.token
    except Exception:
        pass
    return None


def _stage_source(node: ast.AST | None) -> str | None:
    """Best-effort source of the stage (the RHS of a pipe ``BinOp``)."""
    if not isinstance(node, ast.BinOp) or not hasattr(ast, "unparse"):
        return None
    try:
        src = ast.unparse(node.right)
    except Exception:
        return None
    # Statement-block markers (``map[__pyc_block__(5)]``) read better as braces.
    src = re.sub(r"\[__pyc_block__\(\d+\)\]", "{ ... }", src)
    return src


def _enclosing_fork(node: ast.AST | None) -> str | None:
    """If ``node`` sits inside a ``fork[...]``/``parallel[...]`` template, return
    the macro name; else ``None``. Bounded walk up the AST parent chain."""
    if node is None:
        return None
    try:
        import pyccolo as pyc
    except Exception:
        return None
    cur: ast.AST | None = node
    for _ in range(8):
        if cur is None:
            break
        parent = pyc.BaseTracer.containing_ast_by_id.get(id(cur))
        if parent is None:
            break
        if (
            isinstance(parent, ast.Subscript)
            and isinstance(parent.value, ast.Name)
            and parent.value.id in _FORK_MACROS
        ):
            return parent.value.id
        cur = parent
    return None


def _looks_like_leaked_function(exc: BaseException, value: Any) -> bool:
    """True when ``exc`` is the 'a function reached a value position' shape.

    When the piped ``value`` is known we trust it directly (``callable(value)``)
    -- this keeps the hint off *outer* stages that merely propagate a deeper
    function-not-iterable error while piping an ordinary value (e.g. a list).
    Only when the value is unknown (the display-time backstop) do we fall back to
    sniffing the exception message.
    """
    if not isinstance(exc, TypeError):
        return False
    if value is not None:
        return callable(value)
    match = _NOT_A_VALUE_RE.search(str(exc))
    return match is not None and match.group("typ") in _CALLABLE_TYPE_NAMES


def diagnose(
    exc: BaseException,
    *,
    node: ast.AST | None = None,
    value: Any = None,
    func: Any = None,
    generic: bool = False,
) -> str | None:
    """Return a pipescript note for ``exc`` (or ``None`` if nothing useful).

    ``generic=False`` (the runtime apply-site default) emits a note *only* for a
    recognized footgun, so a propagating exception doesn't collect a note at
    every stage it passes through. ``generic=True`` (the display-time backstop,
    which annotates a single node) also reports which stage failed."""
    op = _operator_token(node)
    stage = _stage_source(node)
    if op and stage:
        ctx = f"pipescript: while applying stage `{op} {stage}`"
    elif op:
        ctx = f"pipescript: while applying a `{op}` stage"
    else:
        ctx = "pipescript: while applying a pipeline stage"
    if value is not None:
        ctx += f" to a value of type `{type(value).__name__}`"

    if _looks_like_leaked_function(exc, value):
        fork = _enclosing_fork(node)
        if fork is not None:
            hint = (
                f"a `{fork}[...]` branch must be a *function*, but a value was "
                f"produced here. Use `.>` (compose) instead of `|>` (apply) so "
                f"the branch stays a function -- e.g. `map[...] .> all`, not "
                f"`map[...] |> all`."
            )
        else:
            hint = (
                "a pipescript stage/partial (a function) reached a position "
                "expecting a value. Did you use `|>` (apply) where you meant "
                "`.>` (compose), or leave a `map[...]`/stage un-applied?"
            )
        return f"{ctx}\n  hint: {hint}"

    # No targeted hint: only the backstop reports plain stage context, and only
    # when we actually recovered some pipeline context.
    if generic and (op is not None or value is not None):
        return ctx
    return None


def annotate_pipescript_exception(
    etype: type[BaseException] | None,
    evalue: BaseException | None,
    tb: TracebackType | None,
) -> None:
    """Display-time backstop: enrich ``evalue`` using whatever pipe node we can
    recover from the traceback. No-op if the runtime wrapper already annotated."""
    if evalue is None:
        return
    if getattr(evalue, "_pyc_notes", None):
        return  # the apply-site wrapper already produced a precise note
    try:
        from pipescript.patches.traceback_patch import frame_to_node_mapping
    except Exception:
        return

    node: ast.AST | None = None
    cur = tb
    while cur is not None:
        frame = cur.tb_frame
        candidate = frame_to_node_mapping.get(
            (frame.f_code.co_filename, frame.f_lineno)
        )
        if candidate is not None:
            node = candidate  # keep walking: prefer the deepest mapped frame
        cur = cur.tb_next

    hint = diagnose(evalue, node=node, generic=True)
    if hint:
        add_note(evalue, hint)
