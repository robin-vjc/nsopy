import numpy as np
from methods.quasi_monotone_subgradient_methods import SGMDoubleSimpleAveraging, SGMTripleAveraging
from methods.method_loggers import GenericDualMethodLogger
from examples.analytical_oracles import AnalyticalExampleInnerProblem


def test_DSA_on_analytical_example():
    print('# Test Double Simple Averaging Method on Analytical Example')

    # see definition of AnalyticalExampleInnerProblem for problem and solution statement
    analytical_inner_problem = AnalyticalExampleInnerProblem()

    # it looks like lower gammas give faster convergence, but more oscillations
    GAMMA = 0.5
    dual_method = SGMDoubleSimpleAveraging(analytical_inner_problem.oracle,
                                           analytical_inner_problem.projection_function,
                                           n_constr=analytical_inner_problem.n_constr,
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
                                     n_constr=analytical_inner_problem.n_constr,
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
                                     n_constr=analytical_inner_problem.n_constr,
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