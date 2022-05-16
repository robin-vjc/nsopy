import sys

import numpy as np
import pytest

from nsopy.loggers import EnhancedDualMethodLogger
from nsopy.methods.bundle import CuttingPlanesMethod
from tests.analytical_oracles import (
    AnalyticalExampleInnerProblem, ConstrainedDualAnalyticalExampleInnerProblem, OneDimensionalProblem,
    SecondAnalyticalExampleInnerProblem)


@pytest.mark.skipif('gurobipy' not in sys.modules, reason="requires the Gurobipy library")
def test_cp_method_on_one_dimensional_example():
    print('# Test Cutting Plane Method on One-Dimensional Example')
    analytical_inner_problem = OneDimensionalProblem()

    dual_method = CuttingPlanesMethod(analytical_inner_problem.oracle,
                                      analytical_inner_problem.projection_function,
                                      epsilon=0.01,
                                      sense='min')

    logger = EnhancedDualMethodLogger(dual_method)

    for iteration in range(10):
        print('lambda_k : ', dual_method.lambda_k)
        print('d_k: ', dual_method.d_k)
        dual_method.dual_step()

    # Method should end close to lambda* = 2.25
    # np.testing.assert_allclose(logger.lambda_k_iterates[-1], np.array([1, 1.5]), atol=0.01)
    lambda_star = logger.lambda_k_iterates[-1]
    assert abs(lambda_star[0] - 2.25) <= 0.01


@pytest.mark.skipif('gurobipy' not in sys.modules, reason="requires the Gurobipy library")
def test_cp_method_on_analytical_example():
    print('# Test Cutting Plane Method on Analytical Example (2 ineq)')
    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = AnalyticalExampleInnerProblem()

    dual_method = CuttingPlanesMethod(analytical_inner_problem.oracle,
                                      analytical_inner_problem.projection_function,
                                      dimension=analytical_inner_problem.dimension,
                                      epsilon=0.01,
                                      sense='max')

    logger = EnhancedDualMethodLogger(dual_method)

    for iteration in range(10):
        print(dual_method.lambda_k)
        print(dual_method.d_k)
        dual_method.dual_step()

    # Method should end close to lambda* = [1, [1-1.5]]
    # np.testing.assert_allclose(logger.lambda_k_iterates[-1], np.array([1, 1.5]), atol=0.01)
    lambda_star = logger.lambda_k_iterates[-1]
    assert abs(lambda_star[0] - 1) <= 0.01
    assert lambda_star[1] >= 0.99
    assert lambda_star[1] <= 1.51
    # with value close to dual optimum d*=-1
    np.testing.assert_allclose(logger.d_k_iterates[-1], -0.5, atol=0.02)


@pytest.mark.skipif('gurobipy' not in sys.modules, reason="requires the Gurobipy library")
def test_cp_method_on_second_analytical_example():
    print('# Test Cutting Plane Method on Second Analytical Example (1 eq, 1 ineq)')
    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = SecondAnalyticalExampleInnerProblem()

    dual_method = CuttingPlanesMethod(analytical_inner_problem.oracle,
                                      analytical_inner_problem.projection_function,
                                      dimension=analytical_inner_problem.dimension,
                                      epsilon=0.01,
                                      sense='max')

    logger = EnhancedDualMethodLogger(dual_method)

    for iteration in range(5):
        # print(dual_method.lambda_k)
        # print(dual_method.d_k)
        dual_method.dual_step()

    # Method should end close to lambda* = [1,0]
    np.testing.assert_allclose(logger.lambda_k_iterates[-1], np.array([1, 0]), atol=0.01)
    # with value close to dual optimum d*=-1
    np.testing.assert_allclose(logger.d_k_iterates[-1], -1.02, atol=0.02)


@pytest.mark.skipif('gurobipy' not in sys.modules, reason="requires the Gurobipy library")
def test_cp_method_on_third_analytical_example():
    print('# Test Cutting Plane Method on Constrained Dual Analytical Example')
    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = ConstrainedDualAnalyticalExampleInnerProblem()

    dual_method = CuttingPlanesMethod(analytical_inner_problem.oracle,
                                      analytical_inner_problem.projection_function,
                                      dimension=analytical_inner_problem.dimension,
                                      epsilon=0.01,
                                      sense='max')
    dual_method.set_dual_domain(type='sum to param', param=0.5)
    dual_method.lambda_k = dual_method.projection_function(np.array([-2,2]))

    logger = EnhancedDualMethodLogger(dual_method)

    for iteration in range(5):
        # print(dual_method.lambda_k)
        # print(dual_method.d_k)
        dual_method.dual_step()

    # Method should end close to lambda*, where lambda*[1] = 0.5 - lambda*[0], and 0<= lambda*[0] <= 0.5
    lambda_star = logger.lambda_k_iterates[-1]
    assert 0 <= lambda_star[0] <= 0.5
    assert lambda_star[1] == 0.5 - lambda_star[0]
    # with value close to dual optimum
    np.testing.assert_allclose(logger.d_k_iterates[-1], -1.0, atol=0.01)
