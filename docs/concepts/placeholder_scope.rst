Placeholder scope
=================

A natural question about placeholders (:doc:`/howto/placeholders`) is: given a
``$`` somewhere in an expression, how much of the surrounding code becomes the
body of the induced function? pipescript answers this with two rules.

The rules
---------

#. **If a macro or pipeline step encloses the placeholder**, the induced
   function body is the *smallest* such enclosing macro or pipeline step.
#. **Otherwise**, the body expands to the nearest *chain* of function calls,
   attribute accesses, and/or subscript accesses.

A "chain" is something like ``np.array($).T.astype(int)``: the induced lambda
converts its argument to a NumPy array, transposes it, and casts to ``int`` --
the body expands to include the whole chain, not just ``np.array($)``.

Why it matters
--------------

The rules explain a subtle but important difference between these two sorters:

.. code-block:: python

   # These do NOT do the same thing!
   sorter1 = sorted($, key=$[1])
   sorter2 = sorted($, key=f[$[1]])

``sorter1`` has two ``$`` placeholders that are *not* wrapped by an enclosing
macro, so by rule 2 each induces a separate argument. It is a function of **two**
arguments: a sequence, and a value used (via ``[1]``) to compute the sort key.

``sorter2`` wraps the key expression in ``f[...]``. By rule 1, that ``$`` belongs
to the ``f`` macro's own function, not to the outer one. So ``sorter2`` is a
function of a **single** argument -- a sequence it sorts using the second element
of each item as the key. This is almost always the behavior you want.

The practical takeaway: when a placeholder expression behaves differently from
what you expected, check whether a ``$`` you meant to be "local" is actually
escaping to the outer function -- and wrap it in ``f[...]`` (or name it) to pin
it down.

Related
-------

- :doc:`/howto/placeholders` -- the basics, including named placeholders.
- :doc:`/howto/brace_blocks` -- capturing ``$`` into a local as another way to
  control reuse.
