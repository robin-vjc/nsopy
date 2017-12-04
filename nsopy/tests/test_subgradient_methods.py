from __future__ import print_function

import time

import numpy as np
from nsopy.method_loggers import TemplateMethodLogger, GenericDualMethodLogger, EnhancedDualMethodLogger
from nsopy.subgradient_methods import SubgradientMethod
from nsopy.template_methods import TemplateMethod
from .analytical_oracles import AnalyticalExampleInnerProblem, SecondAnalyticalExampleInnerProblem, ConstrainedDualAnalyticalExampleInnerProblem


def test_templates():
    """ basically make sure that the observer pattern is implemented correctly """
    print('# Test Template Method and Template Logger')
    template_method = TemplateMethod('oracle', 'projection')
    logger = TemplateMethodLogger(template_method)
    for iteration in range(3):
        template_method.dual_step()

    assert logger.x_iterates == [1, 2, 3]

def mock_one_dim_oracle(lambda_k):
    x_k = 1.1
    d_k = 1.2
    diff_d_k = 1.3

    return x_k, d_k, diff_d_k


def mock_projection_function(lambda_k):
    return lambda_k

def test_subgradient_method_sanity_checks():
    print('# Test Subgradient Method Sanity')
    # ensure STEPSIZE_0 has defaults
    # subgradient_method = SubgradientMethod('oracle', 'projection')
    # custom STEPSIZE_0
    subgradient_method = SubgradientMethod(mock_one_dim_oracle,
                                           mock_projection_function,
                                           dimension=1,
                                           stepsize_0=1.5,
                                           stepsize_rule='constant')
    # logging
    logger = GenericDualMethodLogger(subgradient_method)

    # verify initial state
    assert subgradient_method.lambda_k == 0

    for iteration in range(3):
        subgradient_method.dual_step()
        # start: lambda_k = 0
        # it 1: lambda_k = 1.95 = 0 + 1.5 (STEPSIZE_0) * 1.3 (diff_d_k from mock_oracle)
        # it 2: lambda_k = 3.9 = 1.95 + 1.95
        # it 3: lambda_k = 5.85

    for iteration in range(3):
        # default stepsize rule is 1/k
        subgradient_method.dual_step()
        # it 4: lambda_k = 6.3375 = 5.85 + (1.5/4) * 1.3
        # etc.

    print(subgradient_method.desc)

    # verify final states with mock oracle
    assert subgradient_method.x_k == 1.1
    assert subgradient_method.d_k == 1.2

    # print(logger.lambda_k_iterates)
    np.testing.assert_allclose(logger.lambda_k_iterates,
                               np.array([[0], [1.95], [3.9], [5.85], [7.8], [9.75]]),  # [11.7]
                               atol=0.1)


def test_subgradient_method_on_analytical_example():
    print('# Test Subgradient Method on Analytical Example (2 ineq)')
    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = AnalyticalExampleInnerProblem()

    dual_method = SubgradientMethod(analytical_inner_problem.oracle,
                                    analytical_inner_problem.projection_function,
                                    dimension=analytical_inner_problem.dimension)

    logger = GenericDualMethodLogger(dual_method)

    for iteration in range(10):
        # print dual_method.lambda_k
        # print dual_method.d_k
        dual_method.dual_step()

    # Method should end close to lambda*
    np.testing.assert_allclose(logger.lambda_k_iterates[-1], np.array([0.91, 1.]), rtol=1e-2, atol=0)
    # with value close to dual optimum
    np.testing.assert_allclose(logger.d_k_iterates[-1], -0.54, rtol=1e-2, atol=0)


def test_enhanced_logger_on_analytical_example():
    print('# Test Enhanced Logger')
    analytical_inner_problem = AnalyticalExampleInnerProblem()

    dual_method = SubgradientMethod(analytical_inner_problem.oracle,
                                    analytical_inner_problem.projection_function,
                                    dimension=analytical_inner_problem.dimension)

    logger = EnhancedDualMethodLogger(dual_method)

    for iteration in range(10):
        dual_method.dual_step()
        time.sleep(0.01)  # so that we see something in logger timing, otherwise it's all 0, oracle too fast

    # print logger.start_time
    # print logger.iteration_time
    # print logger.oracle_calls
    assert logger.oracle_calls[-1] == 9


def test_subgradient_method_on_second_analytical_example():
    print('# Test Subgradient Method on Second Analytical Example (1 eq, 1 ineq)')
    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = SecondAnalyticalExampleInnerProblem()

    dual_method = SubgradientMethod(analytical_inner_problem.oracle,
                                    analytical_inner_problem.projection_function,
                                    dimension=analytical_inner_problem.dimension)

    logger = GenericDualMethodLogger(dual_method)

    for iteration in range(20):
        # print(dual_method.lambda_k)
        # print(dual_method.d_k)
        dual_method.dual_step()

    # Method should end close to lambda*
    np.testing.assert_allclose(logger.lambda_k_iterates[-1], np.array([1., 0.06]), atol=0.01)
    # with value close to dual optimum
    np.testing.assert_allclose(logger.d_k_iterates[-1], -1.02, atol=0.02)


def test_subgradient_method_on_third_analytical_example():
    print('# Test Subgradient Method on Third Analytical Example (constrained dual). Start at 0, should stay there.')
    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = ConstrainedDualAnalyticalExampleInnerProblem()

    dual_method = SubgradientMethod(analytical_inner_problem.oracle,
                                    analytical_inner_problem.projection_function,
                                    dimension=analytical_inner_problem.dimension)

    # dual_method.lambda_k = dual_method.projection_function(np.array([-1,1]))
    logger = EnhancedDualMethodLogger(dual_method)

    for iteration in range(3):
        # print(dual_method.lambda_k)
        # print(logger.)
        # print(dual_method.d_k)
        dual_method.dual_step()

    # Method should end close to lambda*, where lambda*[1] = 0.5 - lambda*[0], and 0<= lambda*[0] <= 0.5
    lambda_star = logger.lambda_k_iterates[-1]
    assert 0 <= lambda_star[0] <= 0.5
    assert lambda_star[1] == 0.5 - lambda_star[0]
    # with value close to dual optimum
    np.testing.assert_allclose(logger.d_k_iterates[-1], -1.0, atol=0.01)


def test_subgradient_method_on_third_analytical_example_non_zero_start():
    print('# Test Subgradient Method on Third Analytical Example (constrained dual). Start at NON-0.')
    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = ConstrainedDualAnalyticalExampleInnerProblem()

    dual_method = SubgradientMethod(analytical_inner_problem.oracle,
                                    analytical_inner_problem.projection_function,
                                    dimension=analytical_inner_problem.dimension)

    # we set the initial point somewhere not 0
    dual_method.lambda_k = dual_method.projection_function(np.array([-2,2]))
    logger = EnhancedDualMethodLogger(dual_method)

    for iteration in range(25):
        # print(dual_method.lambda_k)
        # print(dual_method.d_k)
        dual_method.dual_step()

    # Method should end close to lambda*, where lambda*[1] = 0.5 - lambda*[0], and 0<= lambda*[0] <= 0.5
    lambda_star = logger.lambda_k_iterates[-1]
    assert 0 <= lambda_star[0] <= 0.5
    assert lambda_star[1] == 0.5 - lambda_star[0]
    # with value close to dual optimum
    np.testing.assert_allclose(logger.d_k_iterates[-1], -1.0, atol=0.01)