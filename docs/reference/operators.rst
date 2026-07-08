Operators
=========

This page is the complete catalog of pipescript's pipe and composition
operators. For a gentler introduction see :doc:`/getting_started/first_pipeline`
and :doc:`/howto/extra_pipes`.

Forward pipe operators
----------------------

The table below lists every forward pipe operator and the vanilla Python it
corresponds to.

.. list-table::
   :header-rows: 1
   :widths: 12 40 48

   * - Operator
     - pipescript syntax
     - Python equivalent
   * - ``|>``
     - ``y = x |> f``
     - ``y = f(x)``
   * - ``|>>``
     - ``x |>> y``
     - ``y = x; y``
   * - ``*|>``
     - ``y = x *|> f`` (``x`` iterable)
     - ``y = f(*x)``
   * - ``**|>``
     - ``y = x **|> f`` (``x`` a dict)
     - ``y = f(**x)``
   * - ``.>``
     - ``h = g .> f``
     - ``h = lambda *a, **kw: g(f(*a, **kw))``
   * - ``*.>``
     - ``h = g *.> f``
     - ``h = lambda *a, **kw: g(*f(*a, **kw))``
   * - ``**.>``
     - ``h = g **.> f``
     - ``h = lambda *a, **kw: g(**f(*a, **kw))``
   * - ``?>``
     - ``y = x ?> f``
     - ``y = None if x is None else f(x)``
   * - ``*?>``
     - ``y = x *?> f`` (``x`` iterable, or ``None``)
     - ``y = None if x is None else f(*x)``
   * - ``**?>``
     - ``y = x **?> f`` (``x`` a dict, or ``None``)
     - ``y = None if x is None else f(**x)``
   * - ``$>``
     - ``g = x $> f``
     - ``g = functools.partial(f, x)``
   * - ``*$>``
     - ``g = x *$> f`` (``x`` iterable)
     - ``g = functools.partial(f, *x)``
   * - ``**$>``
     - ``g = x **$> f`` (``x`` a dict)
     - ``g = functools.partial(f, **x)``

Families at a glance:

- **Apply** (``|>``) -- call the function with the piped value.
- **Assign** (``|>>``) -- write the value to a name, then keep flowing it.
- **Unpack** prefixes -- ``*`` spreads an iterable as positional args, ``**``
  spreads a dict as keyword args.
- **Compose** (``.>``) -- combine two functions instead of applying them; the
  left-hand function runs first.
- **Optional** (``?>``) -- short-circuit to ``None`` when the input is ``None``.
- **Partial** (``$>``) -- build a :func:`functools.partial` rather than calling.

Backward operators
-------------------

Every operator except ``|>>`` has a *backward* variant that swaps which side is
the value and which is the function. ``<|`` is the backward form of ``|>`` (a
low-precedence apply):

.. code-block:: python

   >>> reversed .> list <| [1, 2, 3]
   [3, 2, 1]

Composition power (``.**``)
---------------------------

``.**`` composes a single-argument function with itself a given number of times,
so ``f .** n`` is ``f`` applied ``n`` times:

.. code-block:: python

   >>> 42 |> collatz .** 20   # 42 |> collatz |> collatz |> ... (20 times)

Precedence and associativity
----------------------------

All pipe operators are **left-associative** and share the precedence of ``|``
(bitwise or). Steps run strictly left to right in the order they appear
(including backward pipes). One practical consequence: any pipeline stage that
itself contains a bare ``|`` must be parenthesized so it is not parsed as another
pipe boundary.
