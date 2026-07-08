API reference
=============

This page documents pipescript's importable Python API -- the helper functions
re-exported from the top-level ``pipescript`` namespace, plus the entry point for
registering namespace-block macros. Most day-to-day pipescript is *syntax*
rather than API; you will typically reach for these names only when writing your
own macros or calling a helper directly.

Helper functions
-----------------

These are re-exported from ``pipescript`` (via :mod:`pipescript.api`) and are
also the implementations behind several built-in macros (see
:doc:`macros`).

.. automodule:: pipescript.api.utils
   :members:
   :undoc-members:

.. automodule:: pipescript.api.static_macros
   :members:
   :undoc-members:

Registering namespace-block macros
-----------------------------------

.. autofunction:: pipescript.tracers.macro_tracer.register_namespace_macro
