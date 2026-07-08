Macros and helpers
===================

This page catalogs pipescript's built-in macros and helper utilities. Macros
take a bracketed body (``macro[...]``) or an equivalent brace body
(``macro{...}``; see :doc:`/howto/brace_blocks`); helpers are plain functions you
drop into a pipeline. For the constructors that let you define your own
(``macro[...]``, ``method[...]``), see :doc:`/howto/define_macros`; for autodoc
signatures of the helper functions, see :doc:`api`.

Functional macros
------------------

``map`` / ``filter`` / ``reduce``
  The familiar higher-order functions in bracketed macro form. The function to
  apply goes in the brackets, and ``f`` may be omitted:

  .. code-block:: python

     >>> range(1, 5) |> reduce[$ * $]      # 1*2*3*4
     24
     >>> [1, 2, 3, 4] |> filter[$ % 2 == 0] |> list
     [2, 4]

``f``
  The quick-lambda macro: turns a placeholder expression into a function.
  ``f[$ + $]`` is a two-argument adder. See :doc:`/howto/partials_and_lambdas`.

Side-effect and flow helpers
----------------------------

``do``
  Runs a function for its side effect and forwards the original value
  unchanged (like toolz's ``do``):

  .. code-block:: python

     >>> 2 |> $ + 2 |> do[print] |> $ + 2
     4
     6

``peek``
  Shorthand for ``do[print]`` -- prints the value and returns it. See
  :func:`pipescript.peek`.

``null``
  Swallows its input and returns ``None`` -- useful as a final stage to suppress
  a notebook's automatic rendering of the pipeline result. See
  :func:`pipescript.null`.

``collapse``
  Extracts the single non-null value out of a tuple (as produced by ``fork`` with
  ``when`` branches); raises if there is not exactly one. See
  :func:`pipescript.collapse`.

Branching macros
----------------

``fork`` / ``parallel``
  Apply several functions to the same input and return their results as a tuple.
  ``parallel`` runs the branches concurrently; otherwise it behaves like
  ``fork``:

  .. code-block:: python

     >>> 5 |> fork[$ + 1, $ * 2]
     (6, 10)

``when`` / ``unless``
  ``when`` forwards its input when a predicate holds and otherwise yields a null
  that later stages (e.g. ``collapse``) drop; ``unless`` is its negation.

``otherwise``
  Used as the final branch of a ``fork``/``parallel``; runs only when every
  other branch produced a null.

Iteration macros
----------------

``repeat``
  Repeatedly applies its body until the body yields a null, then returns the last
  non-null value.

``until``
  An alias of ``unless``, read naturally as the stopping condition inside
  ``repeat`` (``repeat[until[$ == 1] .> step]``).

``foreach``
  A *method macro* that runs a block once per item of an iterable; see
  :doc:`/howto/method_macros`.

Concurrency
-----------

``future``
  Schedules its body on another thread and immediately returns a
  :class:`concurrent.futures.Future` for the eventual result:

  .. code-block:: python

     >>> 2 |> future[$ + 2] |> $.result()
     4

Defining your own
-----------------

``macro`` / ``method``
  Constructors that build a new macro from a pipescript template (using ``$$``
  for the argument) or from a Python AST-transforming callable. Full treatment in
  :doc:`/howto/define_macros`.

Other helpers
-------------

pipescript's public helper surface also includes small utilities such as
``push`` / ``pop`` (a value stack), ``lshift`` / ``rshift`` / ``unnest`` (tuple
reshaping), ``replace``, ``write`` / ``read`` (keyed memory), ``expect``,
``memoize``, ``ntimes``, ``once``, and ``context``. See :doc:`api` for their
autodoc signatures.
