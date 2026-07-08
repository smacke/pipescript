Write brace blocks and statement blocks
=======================================

Everywhere a macro accepts a bracketed body (``macro[...]``), pipescript also
accepts a **brace** body (``macro{...}``). For a single expression the two are
interchangeable:

.. code-block:: python

   >>> 5 |> f{ $ + 1 }
   6

Multi-line statement blocks
---------------------------

The payoff of braces is that the body can be a multi-line **statement block** --
loops, conditionals, and intermediate assignments -- which a single bracketed
expression cannot express. The value of the block is its final expression:

.. code-block:: python

   # sum 0..$ with an explicit loop
   >>> 5 |> f{
       acc = 0
       for i in range($):
           acc += i
       acc
   }
   10

Capturing ``$`` into a local
----------------------------

Inside the block, ``$`` keeps its usual meaning (the function's argument). You
can capture it into a local variable to reuse it -- which is handy, because in
the single-expression form each textual ``$`` would otherwise introduce a
*fresh* argument:

.. code-block:: python

   >>> 4 |> f{
       a = $ + 1
       a * a
   }
   25

Conditionals and nested pipelines
----------------------------------

Conditionals work, and pipescript syntax nests freely inside a block, so you can
drop whole pipelines into the body:

.. code-block:: python

   >>> 7 |> f{
       if $ % 2 == 0:
           r = 'even'
       else:
           r = 'odd'
       r
   }
   'odd'

   >>> 10 |> f{
       half = ($ |> f[$ // 2])
       bumped = (half |> f[$ + 1])
       bumped * 100
   }
   600

Brace bodies compose with every macro
--------------------------------------

Statement blocks are not special to ``f`` -- ``do``, ``when``, ``repeat``,
``fork``, and the rest all accept brace bodies. For the tuple-valued macros, a
tuple body inside ``fork{...}`` / ``parallel{...}`` is the multi-function
template, exactly as with the bracketed form:

.. code-block:: python

   >>> 5 |> fork{ $ + 1, $ * 2 }
   (6, 10)

Related
-------

- :doc:`/howto/method_macros` -- ``foreach`` and other method-style macros that
  read naturally with brace blocks.
- :doc:`/reference/macros` -- every macro that accepts a body.
