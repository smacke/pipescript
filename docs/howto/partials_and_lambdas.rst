Build partials and quick lambdas
================================

When a pipeline stage needs a small function, writing out ``lambda`` (or
``functools.partial``) gets tedious. pipescript offers several progressively
terser ways to build one.

Currying with a trailing ``$``
------------------------------

You can curry a function by supplying leading arguments and marking the rest
with ``$``, much like :func:`functools.partial`:

.. code-block:: python

   >>> from functools import reduce
   >>> add_reducer = reduce(lambda x, y: x + y, $, $)
   >>> add_reducer([1, 2, 3], 0)
   6

Partial-call syntax (``$``)
---------------------------

To avoid writing a ``$`` for every trailing argument, prefix the call itself
with ``$`` and simply stop -- pipescript fills in the rest, just like in
`coconut <https://coconut-lang.org/>`_:

.. code-block:: python

   >>> add_reducer = reduce$(lambda x, y: x + y)
   >>> add_reducer([1, 2, 3], 0)
   6
   >>> add_reducer([[1, 2, 3], [4, 5, 6]], [])
   [1, 2, 3, 4, 5, 6]

Because the induced partial keeps the original function's argument defaults, you
can often omit trailing arguments entirely (``reduce``'s initializer here):

.. code-block:: python

   >>> add_reducer = reduce$(lambda x, y: x + y)
   >>> add_reducer([1, 2, 3])
   6

Higher-order macros (``reduce[...]``)
-------------------------------------

Currying ``map``, ``filter``, and ``reduce`` with a function is so common that
pipescript provides bracketed macro forms. The function to curry with goes
between the brackets:

.. code-block:: python

   >>> add_reducer = reduce[lambda x, y: x + y]
   >>> [1, 2, 3] |> add_reducer
   6

Quick lambdas with ``f[...]``
-----------------------------

Writing ``lambda x, y: x + y`` is still verbose. The ``f`` macro turns a
placeholder expression into a lambda:

.. code-block:: python

   >>> add_reducer = reduce[f[$ + $]]
   >>> [1, 2, 3] |> add_reducer
   6

``f`` works on its own, too, and honors named placeholders:

.. code-block:: python

   >>> f[$ + $](2, 3)
   5
   >>> f[$a*$b + $b*$c + $a*$c](2, 3, 4)
   26

Omitting ``f`` inside higher-order macros
-----------------------------------------

Inside a higher-order functional macro you can drop the ``f`` altogether and
write the placeholder expression directly:

.. code-block:: python

   # factorial: 1 * 2 * 3 * 4
   >>> range(1, 5) |> reduce[$ * $]
   24

   # assemble a number from its decimal digits
   >>> [2, 3, 4] |> reduce[10*$ + $]
   234

Related
-------

- :doc:`/howto/brace_blocks` -- when a single expression is not enough.
- :doc:`/reference/macros` -- the full catalog of built-in macros.
