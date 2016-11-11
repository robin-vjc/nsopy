# nsopy -- Non-Smooth Optimization in Python

A set of first-order methods for solving

![optimization problem](notebooks/img/min_opt.png "Non-Smooth Optimization Program")

when
* f(x) is convex, but not necessarily differentiable (has "kinks")
* the set ![X](notebooks/img/XR.png) is convex

**Requirements and Remarks**:
* A first-order oracle for the problem is needed: for a given point x, such an oracle returns the objective 
function value and a valid subgradient; on constrained problems, we also need a valid projection function on ![X](img/XR.png).
See the [basic analytical example](notebooks/Analytical Example.ipynb) for a working setup. 
* Currently, all methods are implemented in Python. Numerical performance is not optimized, but they may
be still useful for quick comparisons or for applications in which the main computational burden is in
evaluating the first order oracle.
* The methods solve the equivalent problem of *maximizing* a concave (non-smooth) function. This is because the main application intended is to 
solve dual problems of minimizations, which naturally arise as maximizations. 
See [basic example on duality](notebooks/Application to Duality.ipynb).

#### Why is this an important class of problems?

There are several contexts in which non-smooth convex optimization programs arise naturally. 
One such context is the lagrangian duality framework. Duality is an approach frequently used to systematically 
take advantage of ``structure'' within difficult optimization programs. 
[Here](notebooks/Application to Duality.ipynb) is quick intro on Lagrangian duality; 
see further below for [more advance applications](#Advanced Applications).

### Basic Usage Examples

* [Simple Analyitical Example](notebooks/Analytical Example.ipynb).

<p align="center">
  <img src="./notebooks/img/solved_ex_1.png" alt="Example" width="50%" href="#"/>
</p>

* [Lagriangian Duality](notebooks/Application to Duality.ipynb). Taking advantage of duality to simplify hard problems. 
Applied to the following Mixed-Integer Linear Program (MILP):

<p align="center">
  <img src="./notebooks/img/primal_problem.png" alt="Example 2" href="#"/>
</p>


### Advanced Applications

* [Decomposition of Stochastic Multistage Integer Programs](https://github.com/robin-vjc/nsopy-stoch).
<p align="center">
  <img src="./notebooks/img/stoch_tree.png" alt="Scenarios Tree" width="60%" href="#"/>
</p>

* Computer Vision (Distributed Computations of Markov Random Fields)

#### Implemented Methods

* standard subgradient method, with constant and 1/k decaying stepsize
* cutting planes (*requires Gurobi*)
* a basic variant of the bundle method (*requires Gurobi*)
* 2x [quasi-monotone methods](http://link.springer.com/article/10.1007/s10957-014-0677-5) (DSA and TA)
* 3x [universal gradient methods](http://link.springer.com/article/10.1007/s10107-014-0790-0) (UPGM, UDGM and UFGM)

**Note**: Cutting Panes and Bundle methods require Gurobi (and the python package ``gurobipy``) to be installed. 
If you are an academic, you can get a free license [here](http://www.gurobi.com/academia/for-universities]). 

### Contributing

Pull requests are very welcome. The [TODO](TODO.txt) contains a number of tasks whose completion would be helpful. 