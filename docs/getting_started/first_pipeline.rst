Your first pipeline
===================

This page assumes you have installed pipescript and run ``%load_ext pipescript``
in your session (see :doc:`installation`).

The pipe operator
-----------------

The heart of pipescript is the **pipe operator**, ``|>``. It takes the value on
its left and feeds it as the argument to the function on its right, so that
``x |> f`` means exactly ``f(x)``:

.. code-block:: python

   >>> tup = (3, 4, 1, 5, 6)
   >>> tup |> sorted |> tuple
   (1, 3, 4, 5, 6)

Read left to right: take ``tup``, sort it, then turn the result into a tuple.
The equivalent vanilla Python, ``tuple(sorted(tup))``, reads inside-out -- you
have to start in the middle and work outward.

Build it up incrementally
-------------------------

Because each stage is an ordinary expression, you can build a pipeline one step
at a time and check the result after every ``|>`` you append. This pairs
perfectly with the way IPython and Jupyter render the value of the last
expression in a cell.

First, run just the first stage:

.. code-block:: python

   >>> tup |> sorted
   [1, 3, 4, 5, 6]

Happy with that? Append the next stage and re-run:

.. code-block:: python

   >>> tup |> sorted |> tuple
   (1, 3, 4, 5, 6)

This tight "append a stage, inspect, repeat" loop is one of the main reasons
pipelines feel good in a notebook.

Passing extra arguments with ``$``
----------------------------------

When the function you are piping into needs more than just the piped value, use
the **placeholder** ``$`` to mark where the piped value goes; every other
argument is written out normally:

.. code-block:: python

   >>> lst = [3, 4, 1, 5, 6]
   >>> lst |> sorted($, reverse=True)
   [6, 5, 4, 3, 1]

Here ``lst`` is substituted for ``$`` and ``reverse=True`` is passed straight
through, so the stage is ``sorted(lst, reverse=True)``. The placeholder is a
whole feature in its own right -- see :doc:`/howto/placeholders`.

Where to go next
----------------

- :doc:`/howto/placeholders` -- everything ``$`` can do.
- :doc:`/tutorials/collatz` -- a longer, end-to-end pipeline.
- :doc:`/reference/operators` -- the other pipe operators (``|>>``, ``*|>``,
  ``.>``, and friends).
