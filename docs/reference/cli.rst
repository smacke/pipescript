Command-line interface
======================

Installing pipescript adds a ``pipescript`` console script. It runs a Python
script under pipescript instrumentation -- the same syntax you get after
``%load_ext pipescript`` in a notebook, but for a ``.py`` file:

.. code-block:: console

   $ pipescript path/to/script.py [args...]

The named script is executed with pipescript's tracers (the pipeline,
placeholder/macro, brace-block, and optional-chaining tracers) enabled for that
file, so it may use ``|>``, ``$``, macros, and the other syntax extensions.
Any additional arguments are forwarded to the script.

.. note::

   Under the hood the ``pipescript`` command is a thin wrapper around pyccolo's
   own command-line runner: it resolves pipescript's tracers, enables them for
   the target file, and delegates to ``pyccolo``'s ``main``. See
   :mod:`pipescript.__main__`.

pipescript remains primarily an interactive tool (see
:doc:`/getting_started/installation`); the CLI is convenient for running a saved
script that happens to use pipescript syntax without converting it back to
vanilla Python first.
