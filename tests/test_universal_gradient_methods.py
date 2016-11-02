from __future__ import print_function
import numpy as np
from methods.universal_gradient_methods import UniversalPGM, UniversalDGM, UniversalFGM
from methods.method_loggers import GenericDualMethodLogger, DualDgmFgmMethodLogger
from examples.analytical_oracles import AnalyticalExampleInnerProblem, SecondAnalyticalExampleInnerProblem, ConstrainedDualAnalyticalExampleInnerProblem

#############
# PGM TESTS #
#############


def test_UPGM_on_analytical_example():
    print('# Test UPGM on Analytical Example')
    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = AnalyticalExampleInnerProblem()

    dual_method = UniversalPGM(analytical_inner_problem.oracle,
                               analytical_inner_problem.projection_function,
                               n_constr=analytical_inner_problem.n_constr,
                               epsilon=0.01)

    logger = GenericDualMethodLogger(dual_method)

    for iteration in range(20):
        # print(dual_method.lambda_k)
        # print(dual_method.d_k)
        dual_method.dual_step()

    # Method should end close to lambda*
    np.testing.assert_allclose(logger.lambda_k_iterates[-1], np.array([1., 1.]), rtol=1e-1, atol=0)
    # with value close to dual optimum
    np.testing.assert_allclose(logger.d_k_iterates[-1], -0.5, rtol=1e-1, atol=0)


def test_UPGM_on_second_analytical_example():
    print('# Test UPGM on Second Analytical Example (1 eq, 1 ineq)')
    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = SecondAnalyticalExampleInnerProblem()

    dual_method = UniversalPGM(analytical_inner_problem.oracle,
                               analytical_inner_problem.projection_function,
                               n_constr=analytical_inner_problem.n_constr,
                               epsilon=0.01)

    logger = GenericDualMethodLogger(dual_method)

    for iteration in range(40):
        # print(dual_method.lambda_k)
        # print(dual_method.d_k)
        # print(logger.L_k_iterates)
        dual_method.dual_step()

    # Method should end close to lambda*
    np.testing.assert_allclose(logger.lambda_k_iterates[-1], np.array([1., 0.]), atol=0.1)
    # with value close to dual optimum
    np.testing.assert_allclose(logger.d_k_iterates[-1], -1, rtol=1e-1, atol=0)


def test_UPGM_on_third_analytical_example_non_zero_start():
    print('# Test UPGM on Third Analytical Example (constrained dual). Start at NON-0.')
    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = ConstrainedDualAnalyticalExampleInnerProblem()

    dual_method = UniversalPGM(analytical_inner_problem.oracle,
                               analytical_inner_problem.projection_function,
                               n_constr=analytical_inner_problem.n_constr)

    # we set the initial point somewhere not 0
    dual_method.lambda_k = dual_method.projection_function(np.array([-2,2]))
    logger = GenericDualMethodLogger(dual_method)

    for iteration in range(5):
        # print(dual_method.lambda_k)
        # print(dual_method.d_k)
        dual_method.dual_step()

    # Method should end close to lambda*, where lambda*[1] = 0.5 - lambda*[0], and 0<= lambda*[0] <= 0.5
    lambda_star = logger.lambda_k_iterates[-1]
    # print(lambda_star)
    assert 0 <= lambda_star[0] <= 0.5
    assert lambda_star[1] == 0.5 - lambda_star[0]
    # with value close to dual optimum
    np.testing.assert_allclose(logger.d_k_iterates[-1], -1.0, atol=0.01)



#############
# DGM TESTS #
#############

def test_UDGM_on_analytical_example():
    print('# Test UDGM on Analytical Example')
    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = AnalyticalExampleInnerProblem()

    dual_method = UniversalDGM(analytical_inner_problem.oracle,
                               analytical_inner_problem.projection_function,
                               n_constr=analytical_inner_problem.n_constr,
                               epsilon=0.01)
    # epsilon=0.5)

    logger = GenericDualMethodLogger(dual_method)

    for iteration in range(20):
        # print(dual_method.lambda_k)
        # print(dual_method.d_k)
        dual_method.dual_step()

    # Method should end close to lambda*
    np.testing.assert_allclose(logger.lambda_k_iterates[-1], np.array([1., 1.]), rtol=1e-1, atol=0)
    # with value close to dual optimum
    np.testing.assert_allclose(logger.d_k_iterates[-1], -0.5, rtol=1e-1, atol=0)


def test_UDGM_on_second_analytical_example():
    print('# Test UDGM on Second Analytical Example (1 eq, 1 ineq)')
    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = SecondAnalyticalExampleInnerProblem()

    dual_method = UniversalDGM(analytical_inner_problem.oracle,
                               analytical_inner_problem.projection_function,
                               n_constr=analytical_inner_problem.n_constr,
                               epsilon=0.01)
    # epsilon=0.5)

    logger = DualDgmFgmMethodLogger(dual_method)

    for iteration in range(40):
        # print(dual_method.lambda_k)
        # print(dual_method.d_k)
        # print(logger.L_k_iterates)
        dual_method.dual_step()

    # Method should end close to lambda*
    np.testing.assert_allclose(logger.lambda_k_iterates[-1], np.array([1., 0.]), atol=0.1)
    # with value close to dual optimum
    np.testing.assert_allclose(logger.d_k_iterates[-1], -1, atol=0.1)


def test_UDGM_on_third_analytical_example_non_zero_start():
    print('# Test UDGM on Third Analytical Example (constrained dual). Start at NON-0.')
    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = ConstrainedDualAnalyticalExampleInnerProblem()

    dual_method = UniversalDGM(analytical_inner_problem.oracle,
                               analytical_inner_problem.projection_function,
                               n_constr=analytical_inner_problem.n_constr,
                               epsilon=0.1)

    # we set the initial point somewhere not 0
    dual_method.lambda_k = dual_method.projection_function(np.array([-2,2]))
    logger = DualDgmFgmMethodLogger(dual_method)

    for iteration in range(25):
        # print(dual_method.lambda_k)
        # print(dual_method.d_k)
        dual_method.dual_step()

    # Method should end close to lambda*, where lambda*[1] = 0.5 - lambda*[0], and 0<= lambda*[0] <= 0.5
    lambda_star = logger.lambda_k_iterates[-1]
    assert 0 <= lambda_star[0] <= 0.5
    assert abs(lambda_star[1] - (0.5 - lambda_star[0])) <= 0.01
    # with value close to dual optimum
    np.testing.assert_allclose(logger.d_k_iterates[-1], -1.0, atol=0.01)


#############
# FGM TESTS #
#############

def test_UFGM_on_analytical_example():
    print('# Test UFGM on Analytical Example')
    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = AnalyticalExampleInnerProblem()

    dual_method = UniversalFGM(analytical_inner_problem.oracle,
                               analytical_inner_problem.projection_function,
                               n_constr=analytical_inner_problem.n_constr,
                               epsilon=0.1)

    logger = GenericDualMethodLogger(dual_method)

    for iteration in range(40):
        # print dual_method.lambda_k
        # print dual_method.d_k
        dual_method.dual_step()

    # Method should end close to lambda*
    np.testing.assert_allclose(logger.lambda_k_iterates[-1], np.array([1., 1.]), rtol=1e-1, atol=0)
    # with value close to dual optimum
    np.testing.assert_allclose(logger.d_k_iterates[-1], -0.5, rtol=1e-1, atol=0)


def test_UFGM_on_second_analytical_example():
    print('# Test UFGM on Second Analytical Example (1 eq, 1 ineq)')
    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = SecondAnalyticalExampleInnerProblem()

    dual_method = UniversalFGM(analytical_inner_problem.oracle,
                               analytical_inner_problem.projection_function,
                               n_constr=analytical_inner_problem.n_constr,
                               epsilon=0.01)
    # epsilon=0.5)

    logger = DualDgmFgmMethodLogger(dual_method)

    for iteration in range(40):
        # print(dual_method.lambda_k)
        # print(dual_method.d_k)
        # print(logger.L_k_iterates)
        dual_method.dual_step()

    # Method should end close to lambda*
    np.testing.assert_allclose(logger.lambda_k_iterates[-1], np.array([1., 0.]), atol=0.1)
    # with value close to dual optimum
    np.testing.assert_allclose(logger.d_k_iterates[-1], -1, atol=0.1)


def test_UFGM_on_third_analytical_example_non_zero_start():
    print('# Test UFGM on Third Analytical Example (constrained dual). Start at NON-0.')
    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = ConstrainedDualAnalyticalExampleInnerProblem()

    dual_method = UniversalFGM(analytical_inner_problem.oracle,
                               analytical_inner_problem.projection_function,
                               n_constr=analytical_inner_problem.n_constr,
                               epsilon=0.01)

    # we set the initial point somewhere not 0
    dual_method.lambda_k = dual_method.projection_function(np.array([-2,2]))
    # print(dual_method.lambda_k)
    logger = DualDgmFgmMethodLogger(dual_method)

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
