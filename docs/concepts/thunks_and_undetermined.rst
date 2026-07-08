Undetermined pipelines and thunks
=================================

Placeholders turn an expression into a function. The same idea, applied to the
*seed* (first stage) of a pipeline, turns the whole pipeline into a function.

Undetermined pipelines
----------------------

Following magrittr, if any placeholders appear in the **first** step of a
pipeline, the pipeline as a whole represents a function -- an *undetermined
pipeline*:

.. code-block:: python

   >>> second_largest_value = $ |> sorted($, reverse=True) |> $[1]
   >>> [3, 8, 6, 5, 1] |> second_largest_value
   6

Since each ``$`` in the seed is a fresh argument, the first step can introduce
more than one, yielding a multi-argument function. Use named placeholders when a
value must be reused within the seed:

.. code-block:: python

   >>> hypotenuse = $a * $a + $b * $b |> $ ** 0.5
   >>> hypotenuse(3, 4)
   5.0

Note that once past the seed, ``$`` has its usual per-stage meaning; it is only
the placeholders *in the seed* that become the pipeline function's parameters.

Thunks: the leading ``|>``
--------------------------

Where ``$ |> ...`` defines a *one*-argument function, a **leading** ``|>``
defines a *zero*-argument one -- a thunk that defers a pipeline until you call
it:

.. code-block:: python

   >>> fetch = |> load() |> transform |> save
   >>> fetch()   # runs load() |> transform |> save now

The pipeline starts from its own seed (the first stage), so no input is needed.
A leading ``|>`` is simply sugar for ``lambda:`` -- it works for any body,
including a single stage (``thunk = |> expensive()``), and it re-runs on every
call (it is deferral, not memoization). Inside the body, ``$`` keeps its usual
meaning in every stage *after* the seed:

.. code-block:: python

   >>> describe = |> 5 |> $ + 1 |> str
   >>> describe()
   '6'

Related
-------

- :doc:`/howto/placeholders` -- how ``$`` induces functions in the first place.
- :doc:`/concepts/placeholder_scope` -- which ``$`` belongs to which function.
