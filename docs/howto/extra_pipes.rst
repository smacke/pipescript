Use the other pipe operators
============================

``|>`` is the pipe you will reach for most, but pipescript offers several
variants for common patterns. This page covers the ones you will use day to day;
:doc:`/reference/operators` has the complete table with the exact Python each one
expands to.

Assignment pipe (``|>>``)
-------------------------

The *assignment pipe* writes the left-hand value into the variable named on the
right, and then evaluates to that same value so the pipeline can continue:

.. code-block:: python

   >>> 2 |> $ + 2 |>> two_plus_two |> $ + 3 |>> two_plus_two_plus_three
   7
   >>> (two_plus_two, two_plus_two_plus_three)
   (4, 7)

Varargs pipe (``*|>``)
----------------------

The *varargs pipe* unpacks the iterable on its left before passing the values as
positional arguments to the function on its right:

.. code-block:: python

   # Add two numbers
   >>> (2, 3) *|> $ + $
   5

A common use is expanding an undetermined pipeline inside a ``map[...]``:

.. code-block:: python

   >>> consecutive_pairs = range(10) |> list |> ($v[::2], $v[1::2]) *|> zip
   >>> consecutive_pairs |> map[$ *|> $ * $] |> list
   [0, 6, 20, 42, 72]

There is a matching ``**|>`` that unpacks a dict as keyword arguments.

Function pipe (``.>``)
----------------------

The *function pipe* **composes** two functions rather than applying them to a
value. The left function runs first (this is the reverse of the mathematical
convention, but it follows the flow of data):

.. code-block:: python

   >>> reverse = reversed .> list
   >>> [1, 2, 3] |> reverse
   [3, 2, 1]

Backward pipes
--------------

Every operator except ``|>>`` has a *backward* variant that flips which side is
the value and which is the function. ``<|`` is the backward form of ``|>`` -- a
low-precedence apply:

.. code-block:: python

   >>> reversed .> list <| [1, 2, 3]
   [3, 2, 1]

Precedence
----------

All pipe operators are left-associative and share the precedence of ``|``
(bitwise or). Consequently, any pipeline stage that itself contains a bare ``|``
must be wrapped in parentheses. See :doc:`/reference/operators` for the full
list, including the optional (``?>``) and partial (``$>``) pipe families.
