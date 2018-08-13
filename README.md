# nsopy -- Non-Smooth Optimization in Python

A set of first-order methods for solving

![optimization problem](img/min_opt.png "Non-Smooth Optimization Program")

when
* f(x) is convex, but not necessarily differentiable
* the set ![X](img/XR.png) is convex

## Installation

The package is pip-installable:

```
>>> pip install nsopy
```

Optional: tests are in ```./nsopy/tests/``` and can be run with the ```py.test``` test runner.


## Basic Usage Example

We want to minimize the non-differentiable function obtained by taking the `max` over a set of functions:
<p align="center">
  <img src="./img/basic_example.png" alt="Example" width="65%" href="#"/>
</p>

It is straightforward to see that the optimum is at `x* = 2.25`; we can solve this optimization problem numerically as follows:
~~~~
from nsopy import SubgradientMethod
from nsopy.method_loggers import GenericMethodLogger
import numpy as np

def oracle(x_k):
    # evaluation of the f_i components at x_k
    fi_x_k = [-2*x_k + 2,  -1.0/3*x_k + 1, x_k - 2]

    f_x_k = max(fi_x_k)  # function value at x_k

    diff_fi = [-1, -1.0/6.0, 1]  # gradients of the components
    max_i = fi_x_k.index(f_x_k)
    diff_f_xk = diff_fi[max_i]  # subgradient at x_k is the gradient of the active function component

    return 0, f_x_k, diff_f_xk

def projection_function(x_k):
    return x_k if x_k is not 0 else np.array([0,])
~~~~
Instantiation of method and logger, solve and print
~~~~
method = SubgradientMethod(oracle=oracle, projection_function=projection_function, stepsize_0=0.1, stepsize_rule='constant', sense='min')
logger = GenericMethodLogger(method)

for iteration in range(200):
    method.step()
~~~~
Result:
~~~~
>>> print(logger.x_k_iterates[-1])

[array([2.2]), array([2.21666667]), array([2.23333333]), array([2.25]), array([2.26666667])]
~~~~


## Available Methods

* **Standard Subgradient Method**

~~~~ 
SubgradientMethod(oracle, projection_function, stepsize_0=1.0, stepsize_rule='1/k', sense='max')
~~~~
Stepsize rules valiable: `stepsize_rule: ['constant', '1/k']`

* **Quasi-Monotone Methods**

Implementation of double simple averaging, and triple averaging methods from Nesterov's paper on [quasi-monotone methods](http://link.springer.com/article/10.1007/s10957-014-0677-5). 

~~~~ 
SGMDoubleSimpleAveraging(oracle, projection_function, gamma=1.0, sense='max')
SGMTripleAveraging(oracle, projection_function, variant=1, gamma=1.0, sense='max'):
~~~~

Variants of `SGMTripleAveraging` available: `variant: [1, 2]`

* **Universal Gradient Methods**

Implementation of Nesterov's [universal gradient methods](http://link.springer.com/article/10.1007/s10107-014-0790-0), primal, dual and fast versions.

~~~~
UniversalPGM(oracle, projection_function, epsilon=1.0, averaging=False, sense='max')
UniversalDGM(oracle, projection_function, epsilon=1.0, averaging=False, sense='max'):        
UniversalFGM(oracle, projection_function, epsilon=1.0, averaging=False, sense='max'):
~~~~

* **Cutting Planes Method**

*Warning*: this method requires `gurobipy`; if you are an academic, you can get a free license [here](http://www.gurobi.com/academia/for-universities]). 

~~~~
CuttingPlanesMethod(oracle, projection_function, epsilon=0.01, search_box_min=-10, search_box_max=10, sense='max')
~~~~

The parameter `epsilon` is the absolute required suboptimality level `|f_k - f*|` used as a stopping criterion. Note that a search box needs to be specified.


* **Bundle Method**

*Warning*: this method requires `gurobipy`; if you are an academic, you can get a free license [here](http://www.gurobi.com/academia/for-universities]). 

Implementation of a basic variant of the bundle method. 

~~~~
BundleMethod(oracle, projection_function, epsilon=0.01, mu=0.5, sense='max'):
~~~~


## Important Remarks

* The first-order oracle must also provide a projection function; [here is a list of cases](img/simple_projections.png) for which 
the projection operation is computationally inexpensive.

* Currently, all methods are implemented in Python. Numerical performance is not optimized, but they may
be still useful for quick comparisons or for applications in which the main computational burden is in
evaluating the first order oracle.


## Advanced Examples

* See [analytical example](./notebooks/AnalyticalExample.ipynb) for a more challenging optimization model.
<p align="center">
  <img src="./img/solved_ex_1.png" alt="Example" width="40%" href="#"/>
</p>


* How to get [approximate solutions to structured MILPs](./notebooks/ApplicationToDuality.ipynb) using Lagrangian duality.
<p align="center">
  <img src="./img/primal_problem.png" alt="Example 2" href="#"/>
</p>


We can also use these methods to decompose stochastic multistage 
mixed integer programs ([preview](https://github.com/robin-vjc/nsopy-stoch)), which in turn allows 
the computation of approximate solutions to these models on distributed environments (e.g., on cloud infrastructure).


### Contributing

Contributions and pull requests are very much welcome. The [TODO](TODO.txt) contains a number of tasks whose completion would be helpful.

## Cite

~~~~
@article{Vujanic2018,
	title={Dual Decomposition of Stochastic Integer Programs: New Results and Experimental Comparison of Solution Methods},
	author={Vujanic, Robin and Esfahani, Peyman Mohajerin},
	journal={TO BE COMPLETED},
	volume={TO BE COMPLETED},
	number={TO BE COMPLETED},
	pages={TO BE COMPLETED},
	year={2018},
	publisher={TO BE COMPLETED}
}
~~~~ 