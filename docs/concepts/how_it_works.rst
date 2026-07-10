How it works
============

pipescript is a *syntax augmentation* of Python: everything you write ends up
running as ordinary Python. Understanding the two-stage transform below explains
both what pipescript can do and why it composes so cleanly.

Two stages
----------

**Stage 1 -- token rewriting.** pipescript first rewrites token spans that are
illegal in Python -- ``|>``, ``*|>``, and the rest -- into legal ones. Most of
the pipe family rewrites to the bitwise-or token ``|``. After this pass the
source is *valid* (though not yet meaningful) Python.

**Stage 2 -- instrumented execution.** The rewritten code is then run in an
*instrumented* fashion. pipescript remembers where each rewrite happened, so
that the resulting :class:`ast.BinOp` node can be associated with the original
``|>`` operator. Instrumentation emits an event for that node, and pipescript's
handler for the event implements the pipe semantics instead of a real bitwise
or.

In short: pipescript rewrites its surface syntax to valid Python, then changes
the *behavior* of specific AST nodes at run time.

The pyccolo backend
-------------------

The heavy lifting is done by `pyccolo <https://github.com/smacke/pyccolo>`_, a
library for declarative, event-driven AST instrumentation. You register handlers
for AST node types (such as :class:`ast.BinOp`); pyccolo instruments those nodes
to emit events, and when the code runs, every registered handler for an event
fires.

The crucial property is **composability**. Because a single event-emitting
transform can feed many independent handlers, layering several instrumentations
usually Just Works -- you do not get the conflicts you would hit hand-writing a
pile of :class:`ast.NodeTransformer` passes that each rewrite the tree. That is
what lets pipescript's own features (pipes, placeholders, macros, brace blocks,
optional chaining) be built as separate tracers that coexist, and it is what
lets pipescript coexist with other pyccolo-based tools like ipyflow.

Consequences
-----------

Because the instrumented code is still Python and stays visually close to the
pipescript source:

- Standard and third-party libraries work unchanged -- it is all just function
  calls underneath.
- Jupyter ergonomics like readable tracebacks and jedi-based autocomplete keep
  working, for the most part. Completion is additionally *type aware*: every
  pipeline in the cell is desugared into the nested calls it denotes before jedi
  sees it, so ``["a", "b"] |> "\n".join($) |> $.`` completes ``str`` methods
  without the pipeline prefix having been executed -- and a name that a pipeline
  binds (via ``=`` or ``|>>``) is typed wherever it is later used. The desugaring
  models the macros and the compose/partial operators as well, down to the
  container type that ``map`` and ``filter`` restore; where it cannot model a
  construct soundly it declines to lower it rather than guess at a type.
- There is a run-time cost when pipescript syntax is actually used, but **none**
  for vanilla Python, even with the extension loaded. See :doc:`performance`.

A Python superset, deliberately scoped
---------------------------------------

pipescript is effectively a Python superset for interactive use. It is *not*
(for now) a general-purpose functional language and is explicitly not intended
for production; it optimizes for simplicity, readability, and predictability
over feature completeness. See :doc:`/getting_started/installation` for the
supported environments.
