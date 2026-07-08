Installation
============

pipescript is published on PyPI. The most common way to install it is from
inside a running IPython or Jupyter session, so that the same session can load
the extension:

.. code-block:: python

   %pip install pipescript
   %load_ext pipescript

The ``%load_ext pipescript`` line is what actually enables the new syntax --
the pipe operators, placeholders, and macros -- in your current session. Without
it, pipescript is installed but dormant.

You can equally install from a normal shell:

.. code-block:: console

   $ pip install pipescript

and then run ``%load_ext pipescript`` once inside IPython/Jupyter.

Runtime dependencies are minimal: `pyccolo
<https://github.com/smacke/pyccolo>`_ (which does the source-to-source
instrumentation -- see :doc:`/concepts/how_it_works`) and ``typing-extensions``.

Supported environments
----------------------

pipescript is currently **only** for interactive Python environments built on
top of IPython -- that is, IPython itself and Jupyter (Notebook, Lab, and
compatible front-ends such as `ipyflow <https://github.com/ipyflow/ipyflow>`_).
It is not a standalone language and has no effect in a plain ``python`` script
run outside of IPython.

It supports **Python 3.9 through 3.14**. Because instrumentation is embedded at
the level of *source code* rather than bytecode, the same syntax works across
that whole range.

Trying it without installing
----------------------------

If you just want to experiment, a full pipescript environment runs entirely in
your browser via JupyterLite -- no local install:

  https://smacke.github.io/pipescript/lab/index.html?path=demo.ipynb

Working from a source checkout
------------------------------

If you are hacking on pipescript itself, clone the repository and install it in
editable mode with the relevant extras:

.. code-block:: console

   $ git clone https://github.com/smacke/pipescript.git
   $ cd pipescript
   $ pip install -e '.[test]'   # test/lint/type-check toolchain
   $ pip install -e '.[docs]'   # Sphinx toolchain for building these docs

You can then build this documentation locally:

.. code-block:: console

   $ make -C docs html

The rendered site lands in ``docs/_build/html``.

Next steps
----------

With pipescript installed and loaded, head to :doc:`/getting_started/first_pipeline`
to build something that runs.
