Welcome to nsopy documentation!
===================================

**nsopy** is a Python library implementing a set of first order methods to solve non-smooth, constrained convex optimization models.

It is applicable to problems of the form

.. math::

    \begin{array}{ll}
    \min & f(x)\\
    \mathrm{s.t.} & x \in \mathbb{X}
    \end{array}

where:

* :math:`f(x)` is convex, but not necessarily differentiable
* :math:`\mathbb{X} \subseteq \mathbb{R}^n` is convex


Installation
------------

.. code-block:: console

   $ pip install nsopy


.. _example:

Example
-------

We seek to minimize the following piece-wise affine function:

.. math::

    \begin{array}{ll}
    \min\limits_x & \max_i f_i(x)\\
    \mathrm{s.t.} & x \geq 0
    \end{array}

with

.. math::

    \begin{array}{ll}
    f_1(x) = -2x + 2\\
    f_2(x) = -\frac{1}{3}x + 1\\
    f_3(x) = x - 2
    \end{array}

.. image:: docs/example_graph.png

**Constraints:** In the minimization problem we require that :math:`x \geq 0`.
To enable **nsopy** to satisfy this, we need to supply it with a **projection function**: given a point :math:`x` that
does not necessarily satisfy :math:`x \geq 0`, it returns the *closest* (in :math:`\ell_2` sense) point that does.

For this example:

.. code-block:: python

    def projection_function(x_k):
        return np.maximum(x_k, 0)

Check out the :doc:`usage` section for further information, including how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   usage
   api