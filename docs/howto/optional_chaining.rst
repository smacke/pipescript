Optional chaining and nullish coalescing
=========================================

pipescript provides TypeScript-style optional chaining and nullish coalescing,
plus a permissive attribute-access operator.

Optional chaining (``?.``)
--------------------------

``a?.b`` short-circuits to ``None`` when ``a`` is ``None``. The short-circuit
propagates through the rest of the chain, so ``a?.b.c.d().e`` resolves to
``None`` whenever ``a`` is ``None``, and ``a?.()`` calls ``a`` only when it is
not ``None``.

Nullish coalescing (``??``)
---------------------------

``a ?? obj`` evaluates to ``obj`` **only** when ``a`` is ``None``; for any other
value -- including falsy ones like ``""``, ``0``, ``False``, or ``[]`` -- it
evaluates to ``a``. Like boolean ``or``, ``??`` is lazy: the right-hand side is
not evaluated when the left-hand side is non-``None``.

Permissive attribute access (``.?``)
------------------------------------

Unlike JavaScript, Python raises :exc:`AttributeError` for a missing attribute
rather than resolving it to ``undefined``. When you want JavaScript-like
permissive access, use the *permissive chaining operator* ``.?`` (the ``?``
comes **after** the dot):

.. code-block:: python

   a.?b   # equivalent to getattr(a, "b", None)

Note the operators are distinct and compose. ``.?`` alone does not guard the
rest of the chain: if ``a.?b`` resolves to ``None``, then ``a.?b.c`` still
raises :exc:`AttributeError`. To be safe on both counts, combine permissive
access with optional chaining:

.. code-block:: python

   a.?b?.c
