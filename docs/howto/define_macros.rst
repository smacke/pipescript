Define your own macros
======================

Beyond the built-ins (``map``, ``filter``, ``fork``, ``when``, ...) pipescript
lets you define your own macros. Define one in a cell and it is registered for
use in the cells that follow. There are two ways to write one: with the macro
DSL, or with a custom node transformer.

With the macro DSL
------------------

The ``macro[...]`` and ``method[...]`` constructors build a macro out of a
pipescript *template*, using ``$$`` to mark where the macro's argument should be
spliced in. For example, a ``switch`` macro that runs its branches as a ``fork``
and collapses to the single non-null result:

.. code-block:: python

   >>> switch = macro[fork[$$] .> collapse]
   >>> 1 |> switch[when[$ == 0] .> $ + 1, when[$ == 1] .> $ - 1]
   0
   >>> 0 |> switch[when[$ == 0] .> $ + 1, when[$ == 1] .> $ - 1]
   1

``$$`` may be named (``$$v``) when the same argument is spliced in more than one
place, and a macro can take several arguments by listing their names after the
template:

.. code-block:: python

   # swap a pair of values
   >>> flip = macro[($$b, $$a), a, b]
   >>> flip[0, 1]
   (1, 0)

Macros can be defined in terms of other macros (expansion is recursive), and
``method[...]`` defines a macro you invoke as a method on its first argument --
exactly how the built-in ``foreach`` is defined
(``method[$$ |> map[do[$$]] |> list]``; see :doc:`method_macros`).

With a custom node transformer
------------------------------

When a template is not expressive enough, hand ``macro[...]`` a Python callable
instead. pipescript looks the name up in your namespace and, at expansion time,
calls it with the *AST node* written inside the brackets; whatever expression
you return is spliced in (it may itself be pipescript, e.g. an ``f[...]`` quick
lambda). Because the macro sees its argument's syntax, it can do things a
template cannot -- such as reading the argument's source text.

Here is a ``dbg``-style macro (à la Rust's ``dbg!``) that prints an expression
next to its value and then evaluates to it:

.. code-block:: python

   import ast

   def show(node):
       # `node` is the AST of whatever appears inside `dbg[...]`. Read its source
       # text and emit `(print("<src> =", <expr>), <expr>)[1]` -- which prints the
       # labeled value, then evaluates to it.
       label = ast.Constant(ast.unparse(node) + " =")
       printed = ast.Call(ast.Name("print", ast.Load()), [label, node], [])
       pair = ast.Tuple([printed, node], ast.Load())
       result = ast.Subscript(pair, ast.Constant(1), ast.Load())
       return ast.fix_missing_locations(ast.copy_location(result, node))

   dbg = macro[show]

.. code-block:: python

   >>> x = 5
   >>> dbg[x * 2 + 1]
   x * 2 + 1 = 11
   11
   >>> dbg[x * 2 + 1] |> $ + 100   # works mid-pipeline too
   x * 2 + 1 = 11
   111

The callable can equally be a class implementing ``__call__(self, node)`` when
it needs to carry state while inspecting the node.

Namespace-block macros
----------------------

pipescript also supports *namespace-block macros*, which run a brace block and
harvest its top-level assignments into a dict before handing them to a builder
function. Register one with ``register_namespace_macro`` from
:mod:`pipescript.tracers.macro_tracer`.

Related
-------

- :doc:`/concepts/how_it_works` -- what "expansion time" means and why macros
  compose.
- :doc:`/reference/api` -- the registration entry points.
