Tutorial: the Collatz sequence
==============================

This tutorial builds a `Collatz sequence <https://en.wikipedia.org/wiki/Collatz_conjecture>`_
generator from scratch. Along the way you will meet ``when``, ``unless``,
``fork``, ``collapse``, ``otherwise``, ``peek``, the composition-power operator
``.**``, and finally ``repeat`` / ``until``. It assumes you have read
:doc:`/getting_started/first_pipeline`.

The rule is: if ``n`` is even, halve it; if it is odd, compute ``3n + 1``; stop
at ``1``.

Branching with ``fork`` and ``collapse``
-----------------------------------------

``fork`` applies several functions to the same input and returns their results
as a tuple. ``when`` forwards its input when a condition holds and otherwise
yields a null that later stages drop. ``collapse`` extracts the single non-null
value out of a tuple. Put together, they express "do exactly one of these
branches":

.. code-block:: python

   >>> collatz = when[$ != 1] .> fork[
       when[$ % 2 == 0] .> $ // 2,
       when[$ % 2 == 1] .> $ * 3 + 1,
   ] .> collapse .> peek

The leading ``when[$ != 1]`` stops the sequence at ``1``. ``peek`` prints the
value as it flows past (and returns it unchanged) so we can watch the sequence.

``unless`` and ``otherwise``
----------------------------

Writing the second condition as the negation of the first is repetitive.
``unless`` is the opposite of ``when``:

.. code-block:: python

   >>> collatz = when[$ != 1] .> fork[
       when[$ % 2 == 0] .> $ // 2,
       unless[$ % 2 == 0] .> $ * 3 + 1,
   ] .> collapse .> peek

Better still, ``fork`` accepts an ``otherwise`` branch as its last entry, which
runs only when every other branch produced a null:

.. code-block:: python

   >>> collatz = when[$ != 1] .> fork[
       when[$ % 2 == 0] .> $ // 2,
       otherwise[$ * 3 + 1],
   ] .> collapse .> peek

The concise form
----------------

For a rule this simple, a single quick lambda with a ternary is clearest of all:

.. code-block:: python

   >>> collatz = when[$ != 1] .> f[$v // 2 if $v % 2 == 0 else $v * 3 + 1] .> peek

Note the named placeholder ``$v``: the argument is referenced twice, so it must
be the *same* argument (see :doc:`/howto/placeholders`).

Iterating: ``.**`` and ``repeat`` / ``until``
----------------------------------------------

We do not want to write ``42 |> collatz |> collatz |> ...`` by hand. The
composition-power operator ``.**`` composes a single-argument function with
itself ``n`` times:

.. code-block:: python

   >>> 42 |> collatz .** 20
   21
   64
   32
   16
   8
   4
   2
   1

But guessing the exponent is awkward. ``repeat`` keeps applying its body until
the body yields a null, and ``until`` (an alias of ``unless``) supplies that
stopping condition:

.. code-block:: python

   >>> collatz = f[$v // 2 if $v % 2 == 0 else $v * 3 + 1]
   >>> 42 |> repeat[until[$ == 1] .> collatz .> peek] |> null
   21
   64
   32
   16
   8
   4
   2
   1

The trailing ``|> null`` swallows ``repeat``'s return value so the notebook does
not also render it (see :func:`pipescript.null`).

Where to go next
----------------

- :doc:`/reference/macros` -- the reference for every macro used here.
- :doc:`parsing` -- a second tutorial, on collapsing nested calls.
