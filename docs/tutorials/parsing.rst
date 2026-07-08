Tutorial: untangling nested calls
==================================

pipescript was written alongside `Advent of Code 2025
<https://adventofcode.com/2025>`_, where the input-parsing step is often a
tangle of nested calls. This tutorial takes one such tangle and rewrites it as a
readable pipeline. It assumes you have read :doc:`/getting_started/first_pipeline`
and :doc:`/howto/placeholders`.

The starting point
------------------

Consider computing, across a collection of arrays, the maximum finite absolute
value (with a floor of ``1.0``):

.. code-block:: python

   import numpy as np

   result = max(
       np.max(np.abs(array[np.isfinite(array)]), initial=1.0)
       for array in arrays
   )

It is hard to read: you have to work inside-out, and it is genuinely ambiguous at
a glance which call the ``initial=1.0`` keyword belongs to.

Rewriting as a pipeline
-----------------------

The same computation reads top-to-bottom in pipescript. We map each array
through a small inner pipeline, then take the ``max``:

.. code-block:: python

   result = arrays |> map[$
     |> $array[np.isfinite($array)]
     |> np.abs
     |> np.max($, initial=1.0)
   ] |> max

Read it as a recipe:

#. ``arrays |> map[...]`` -- run the bracketed pipeline once per array.
#. ``$array[np.isfinite($array)]`` -- keep only the finite entries. The named
   placeholder ``$array`` is used twice, so both refer to the same array (an
   unnamed ``$`` would introduce two separate arguments -- see
   :doc:`/concepts/placeholder_scope`).
#. ``|> np.abs`` -- take absolute values.
#. ``|> np.max($, initial=1.0)`` -- reduce to a maximum, with the floor made
   unambiguous by the explicit ``$``.
#. ``|> max`` -- the maximum across all arrays.

Why this is nicer in a notebook
-------------------------------

Beyond readability, the pipeline form lets you verify each step interactively:
run ``arrays |> map[$ |> $array[np.isfinite($array)]] |> list`` on its own to
confirm the filtering, then append the next stage, and so on -- the incremental
workflow from :doc:`/getting_started/first_pipeline`.

More examples
-------------

The author's full set of AoC 2025 solutions -- many built around pipescript
input parsing -- lives at https://github.com/smacke/aoc2025. The `day 6 solution
<https://github.com/smacke/aoc2025/blob/main/aoc6.ipynb>`_ pushes the syntax to
its limits (optimized for pipescript, not readability). See also
:doc:`/examples`.
