Welcome to nsopy documentation!
===================================

**nsopy** is a Python library implementing a set of first order methods to solve non-smooth, constrained convex optimization models.

It is applicable to problems of the form

.. math::

    \begin{eqnarray}
    \min & f(x)\\
    \mathrm{s.t.} & x \in \mathbb{X}
    \end{eqnarray}

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

We seek to minimize the following piece-wise affine function:

.. math::

    \begin{eqnarray}
    \min_x & \max_i f_i(x)\\
    \mathrm{s.t.} & x \geq 0
    \end{eqnarray}

with

.. math::

    \begin{aligned}
    f_1(x) = -2x + 2\\
    f_2(x) = -\frac{1}{3}x + 1\\
    f_3(x) = x - 2
    \end{aligned}

In the minimization problem we require our solutions to satisfy the constraint :math:`x \geq 0`.
To enable `nsopy` to satisfy this, we need to provide a **projection function**: given a point $x$ that
does not necessarily satisfy :math:`x \geq 0`, it returns the *closest* (in :math:`\ell_2` sense) point that does.

This is how we can construct the projection function in this case:

.. code-block:: python

    def projection_function(x_k):
        if x_k is 0:
            return np.array([0,])
        else:
            return np.maximum(x_k, 0)

Check out the :doc:`usage` section for further information, including how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   usage
   api