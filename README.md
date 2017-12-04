# nsopy -- Non-Smooth Optimization in Python

A set of first-order methods for solving

![optimization problem](./notebooks/img/min_opt.png "Non-Smooth Optimization Program")

when
* f(x) is convex, but not necessarily differentiable (has "kinks")
* the set ![X](./notebooks/img/XR.png) is convex.

## Usage Examples

* See the [basic analytical example](./notebooks/Analytical Example.ipynb) for a complete working setup.
<p align="center">
  <img src="./notebooks/img/solved_ex_1.png" alt="Example" width="30%" href="#"/>
</p>

* How to get approximate solutions to structured MILPs using [Lagriangian Duality](./notebooks/Application to Duality.ipynb).

<p align="center">
  <img src="./notebooks/img/primal_problem.png" alt="Example 2" href="#"/>
</p>

* Example showing how to [decompose stochastic multistage mixed integer programs](https://github.com/robin-vjc/nsopy-stoch).
<p align="center">
  <img src="./notebooks/img/stoch_tree.png" alt="Scenarios Tree" width="30%" href="#"/>
</p>

* Computer Vision (Distributed Computations of Markov Random Fields)


#### Currently Implemented Methods

* standard subgradient method, with constant and 1/k decaying stepsize
* cutting planes (*requires Gurobi*)
* a basic variant of the bundle method (*requires Gurobi*)
* 2x [quasi-monotone methods](http://link.springer.com/article/10.1007/s10957-014-0677-5) (DSA and TA)
* 3x [universal gradient methods](http://link.springer.com/article/10.1007/s10107-014-0790-0) (UPGM, UDGM and UFGM)

**Note**: Cutting Panes and Bundle methods require Gurobi (and the python package ``gurobipy``) to be installed. 
If you are an academic, you can get a free license [here](http://www.gurobi.com/academia/for-universities]). 

**Requirements and Important Remarks:**

* The first-order oracle must also provide a projection function; [here is a list of cases](notebooks/img/simple_projections.png) for which 
projection is computationally inexpensive.

* Currently, all methods are implemented in Python. Numerical performance is not optimized, but they may
be still useful for quick comparisons or for applications in which the main computational burden is in
evaluating the first order oracle.


### Contributing

Pull requests are very welcome. The [TODO](TODO.txt) contains a number of tasks whose completion would be helpful. 