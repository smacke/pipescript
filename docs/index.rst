pipescript
==========

**pipescript** is an IPython extension that brings a pipe operator (``|>``) and
powerful placeholder and macro-expansion syntax to IPython and Jupyter. If you
have used the `magrittr <https://magrittr.tidyverse.org/>`_ package for R, you
will feel right at home.

Here is the smallest interesting program. Load the extension, then pipe a value
through a couple of functions:

.. code-block:: python

   %load_ext pipescript

   tup = (3, 4, 1, 5, 6)
   tup |> sorted |> tuple   # -> (1, 3, 4, 5, 6)

The pipe operator makes the flow of execution follow the natural left-to-right
order in which you read and write code, and -- because each stage is just a
Python expression -- you can build a pipeline incrementally, inspecting the
result after every ``|>`` you append.

.. note::

   pipescript is aimed at interactive, scratchpad-style work in IPython and
   Jupyter. It is not (yet) a general-purpose functional language on top of
   Python, and is explicitly **not** intended for production use. See
   :doc:`/concepts/how_it_works` for what that means in practice.

✨ You can also `try pipescript in your browser
<https://smacke.github.io/pipescript/lab/index.html?path=demo.ipynb>`_ ✨ -- no
install required.

Where to go from here
---------------------

This documentation is organized in four tracks, following the `Diátaxis
<https://diataxis.fr/>`_ framework:

- **Getting started** -- install pipescript and run your first pipeline.
- **Tutorials** -- learning-oriented, end-to-end builds you follow start to finish.
- **How-to guides** -- short, focused recipes for a specific task.
- **Concepts** -- the mental model: how the syntax works and why.
- **Reference** -- the precise catalog of every operator, macro, and helper.

If you are brand new, read :doc:`/getting_started/installation` then
:doc:`/getting_started/first_pipeline`. If you want to understand *why* it works,
start with :doc:`/concepts/how_it_works`. If you are looking something up, jump to
:doc:`/reference/operators` or :doc:`/reference/macros`.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started/installation
   getting_started/first_pipeline

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/collatz
   tutorials/parsing

.. toctree::
   :maxdepth: 2
   :caption: How-to guides

   howto/placeholders
   howto/partials_and_lambdas
   howto/brace_blocks
   howto/method_macros
   howto/define_macros
   howto/extra_pipes
   howto/optional_chaining

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   concepts/how_it_works
   concepts/placeholder_scope
   concepts/thunks_and_undetermined
   concepts/performance

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/operators
   reference/macros
   reference/cli
   reference/api

.. toctree::
   :maxdepth: 1
   :caption: More

   examples
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
