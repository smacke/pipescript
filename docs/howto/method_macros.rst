Use method macros
==================

Some macros read most naturally as *methods* on the value they consume, rather
than as a function applied to it. pipescript ships one such built-in,
``foreach``, and lets you define your own (see :doc:`define_macros`).

``foreach``
-----------

``foreach`` runs a block once per item of an iterable. Inside the block, ``$``
is the current element:

.. code-block:: python

   >>> out = []
   >>> range(4).foreach{
       v = $
       out.append(v * v)
   }
   >>> out
   [0, 1, 4, 9]

Because the block reads and mutates variables from the enclosing scope,
``foreach`` is a convenient drop-in for an imperative ``for`` loop.

As a pipe stage
---------------

A method macro can also be used as a pipe stage by writing it on the ``$``
receiver, in which case the receiver *is* the piped value:

.. code-block:: python

   >>> out = []
   >>> range(4) |> $.foreach[out.append($ ** 2)]
   >>> out
   [0, 1, 4, 9]

Under the hood
--------------

``foreach`` is nothing special -- it is itself a built-in *dynamic macro*,
defined as::

   method[$$ |> map[do[$$]] |> list]

The next page, :doc:`define_macros`, shows how ``method[...]`` and ``macro[...]``
work so you can build your own.
