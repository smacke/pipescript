Use placeholders (``$``)
========================

The placeholder ``$`` stands in for a function argument and *induces the
creation of a function*. It is what makes the pipe operator so expressive, but
it works anywhere, not just inside a pipeline.

The basic idea
--------------

Inside a pipe stage, ``$`` marks where the piped value should be substituted:

.. code-block:: python

   >>> lst = [3, 4, 1, 5, 6]
   >>> lst |> sorted($, reverse=True)
   [6, 5, 4, 3, 1]

Away from a pipeline, an expression containing ``$`` evaluates to a function.
So you can name it and call it directly:

.. code-block:: python

   lst = [3, 4, 1, 5, 6]
   reverse_sorter = sorted($, reverse=True)

   # These are equivalent:
   print(reverse_sorter(lst))
   lst |> reverse_sorter |> print

Each ``$`` is a new argument
----------------------------

Every occurrence of ``$`` introduces a *new* positional argument, in
left-to-right order. So ``sorted($, reverse=$)`` is a function of **two**
arguments:

.. code-block:: python

   import random

   lst = [3, 4, 1, 5, 6]
   sorter = sorted($, reverse=$)
   reverse = random.random() < 0.5

   # Equivalent:
   print(sorter(lst, reverse))
   lst |> sorter($, reverse) |> print

Named placeholders
------------------

When you need to reference the *same* argument more than once, give the
placeholder a name by writing an identifier right after ``$``. Every occurrence
of a given name refers to the same argument:

.. code-block:: python

   # Pair even-index entries from a range with their adjacent odd-index entry
   >>> range(6) |> list |> zip($v[::2], $v[1::2]) |> list
   [(0, 1), (2, 3), (4, 5)]

Any name works; the important thing is that both slices use ``$v``. Had they
used different names, pipescript would have induced a two-argument function
instead of one.

Placeholders can appear anywhere
--------------------------------

``$`` is not limited to being a call argument -- it can appear in the middle of
an attribute or subscript chain, and pipescript will figure out how much of the
surrounding expression to include in the induced function:

.. code-block:: python

   # Sort a list, then find the index of a value
   >>> lst = [3, 4, 1, 5, 6]
   >>> lst |> sorted |> $.index(3)
   1

Exactly how much of the surrounding code gets pulled into the function body is
governed by a small set of scoping rules. When a placeholder expression does
something subtly different from what you expected, that is almost always the
cause -- see :doc:`/concepts/placeholder_scope`.

Related
-------

- :doc:`/howto/partials_and_lambdas` -- terser ways to build small functions.
- :doc:`/concepts/thunks_and_undetermined` -- pipelines that are themselves functions.
