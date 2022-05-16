Welcome to nsopy documentation!
===================================

**nsopy** is a Python library implementing a set of first order methods to solve non-smooth, convex optimization models.

It is applicable to problems of the form

.. math::

    \begin{aligned}
    \min & f(x)\\
    \mathrm{s.t.} & x \in \mathbb{X}
    \end{aligned}

where:

* :math:`f(x)` is convex, but not necessarily differentiable
* :math:`\mathbb{X} \subseteq \mathbb{R}^n` is convex


Installation
------------

.. code-block:: console

   $ pip install nsopy


.. _quickstart:

Quick Start
-----------

We seek to minimize the following piecewise affine function:

.. math::

    \begin{aligned}
    \min & f(x)\\
    \mathrm{s.t.} & x \in \mathbb{X}
    \end{aligned}


Check out the :doc:`usage` section for further information, including how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   usage
   api