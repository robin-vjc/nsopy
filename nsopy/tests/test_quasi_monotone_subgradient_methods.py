from __future__ import print_function

import numpy as np
from nsopy.method_loggers import GenericDualMethodLogger
from nsopy.quasi_monotone_subgradient_methods import SGMDoubleSimpleAveraging, SGMTripleAveraging
from nsopy.tests.analytical_oracles import AnalyticalExampleInnerProblem, BertsekasCounterExample


def test_DSA_on_analytical_example():
    print('# Test Double Simple Averaging Method on Analytical Example')

    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = AnalyticalExampleInnerProblem()

    # it looks like lower gammas give faster convergence, but more oscillations
    GAMMA = 0.5
    dual_method = SGMDoubleSimpleAveraging(analytical_inner_problem.oracle,
                                           analytical_inner_problem.projection_function,
                                           dimension=analytical_inner_problem.dimension,
                                           gamma=GAMMA)
    logger = GenericDualMethodLogger(dual_method)

    for iteration in range(20):
        # print dual_method.lambda_k
        # print dual_method.d_k
        dual_method.dual_step()

    # Method should end close to lambda*
    # np.testing.assert_allclose(logger.lambda_k_iterates[-1], np.array([1., 1.]), rtol=1e-1, atol=0)
    assert 0.95 <= logger.lambda_k_iterates[-1][0] <= 1.05  # first coordinate should be ~1.0
    assert 0.95 <= logger.lambda_k_iterates[-1][1] <= 1.55  # second coordinate should be 1.0 <= coord <= 1.5
    # with value close to dual optimum
    np.testing.assert_allclose(logger.d_k_iterates[-1], -0.5, rtol=1e-1, atol=0)


def test_TSA_variant_1_on_analytical_example():
    print('# Test Triple Averaging Method, variant 1, on Analytical Example')

    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = AnalyticalExampleInnerProblem()

    # it looks like lower gammas give faster convergence, but more oscillations
    GAMMA = 0.5
    dual_method = SGMTripleAveraging(analytical_inner_problem.oracle,
                                     analytical_inner_problem.projection_function,
                                     dimension=analytical_inner_problem.dimension,
                                     variant=1,
                                     gamma=GAMMA)
    logger = GenericDualMethodLogger(dual_method)

    for iteration in range(20):
        # print dual_method.lambda_k
        # print dual_method.d_k
        dual_method.dual_step()

    # Method should end close to lambda*
    # np.testing.assert_allclose(logger.lambda_k_iterates[-1], np.array([1., 1.]), rtol=1e-1, atol=0)
    assert 0.95 <= logger.lambda_k_iterates[-1][0] <= 1.05  # first coordinate should be ~1.0
    assert 0.95 <= logger.lambda_k_iterates[-1][1] <= 1.55  # second coordinate should be 1.0 <= coord <= 1.5
    # with value close to dual optimum
    np.testing.assert_allclose(logger.d_k_iterates[-1], -0.5, rtol=1e-1, atol=0)


def test_TSA_variant_2_on_analytical_example():
    print('# Test Triple Averaging Method, variant 2, on Analytical Example')

    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = AnalyticalExampleInnerProblem()

    # it looks like lower gammas give faster convergence, but more oscillations
    GAMMA = 1.0
    dual_method = SGMTripleAveraging(analytical_inner_problem.oracle,
                                     analytical_inner_problem.projection_function,
                                     dimension=analytical_inner_problem.dimension,
                                     variant=2,
                                     gamma=GAMMA)
    logger = GenericDualMethodLogger(dual_method)

    for iteration in range(60):
        # print dual_method.lambda_k
        # print dual_method.d_k
        dual_method.dual_step()

    # Method should end close to lambda*
    # np.testing.assert_allclose(logger.lambda_k_iterates[-1], np.array([1., 1.]), rtol=1e-1, atol=0)
    assert 0.95 <= logger.lambda_k_iterates[-1][0] <= 1.05  # first coordinate should be ~1.0
    assert 0.95 <= logger.lambda_k_iterates[-1][1] <= 1.55  # second coordinate should be 1.0 <= coord <= 1.5
    # with value close to dual optimum
    np.testing.assert_allclose(logger.d_k_iterates[-1], -0.5, rtol=1e-1, atol=0)


def test_DSA_on_Bertsekas_example():
    print('# DSA on Bertsekas Example')

    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = BertsekasCounterExample()

    # it looks like lower gammas give faster convergence, but more oscillations
    GAMMA = 0.5
    dual_method = SGMDoubleSimpleAveraging(analytical_inner_problem.oracle,
                                           analytical_inner_problem.projection_function,
                                           dimension=analytical_inner_problem.dimension,
                                           gamma=0.5)
    logger = GenericDualMethodLogger(dual_method)

    # move initial point
    lambda_0 = np.array([2, 2])
    dual_method.lambda_k = lambda_0

    for iteration in range(20):
        # print(dual_method.lambda_k)
        # print(dual_method.d_k)
        dual_method.dual_step()
        # print(logger.d_k_iterates)
        # print(logger.lambda_k_iterates)

    # Method should end close to lambda*
    assert -3 <= logger.lambda_k_iterates[-1][0] <= -2.2  # first coordinate should be ~3.0
    assert 0 <= logger.lambda_k_iterates[-1][1] <= 0.5  # second coordinate should be ~0
    # with value close to dual optimum
    np.testing.assert_allclose(logger.d_k_iterates[-1], 23.15, rtol=1e-1, atol=0)