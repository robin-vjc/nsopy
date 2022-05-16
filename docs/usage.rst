Usage
=====

.. _installation:

Installation
------------

To use nsopy, first install it using pip:

.. code-block:: console

   (.venv) $ pip install lumache

Quick Start
-----------

To retrieve a list of random ingredients,
you can use the ``nsopy.loggers.GenericMethodLogger`` function:

.. autofunction:: nsopy.utils.invert_oracle_sense

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`nsopy.loggers.GenericMethodLogger`
will raise an exception.

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']
