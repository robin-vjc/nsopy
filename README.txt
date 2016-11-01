# nsopy -- Non-Smooth Optimization for Python

# About

This repo contains Python implementations of several first-order methods for solving the dual problems arising in the
context of **dual decomposition** of
 * Stochastic (Mixed Integer) Optimization Models
 * Markov Random Fields (with discrete labels)

Methods' efficacy is assessed in terms of number of necessary oracle calls.
In particular we are interested in how the more recently proposed methods
* ??
* ??
stack against traditional options like subgradient, cutting plane and bundle methods.
See ???PAPER DRAFT for a more detailed explanation.

# Usage

To see usage examples, please check the iPython notebooks in `./notebooks/`. ???LINK

Non-pip libraries required to run the notebooks locally:
* ???LINK gurobipy
* ???LINK opengm (with Python interface)
