# Centralize instantiation of dual methods. Useful in particular since the different
# method classes have slightly different instantiation parameters.

from nsopy.methods.subgradient import SubgradientMethod
from nsopy.methods.universal import UniversalPGM, UniversalDGM, UniversalFGM
from nsopy.methods.quasi_monotone import SGMDoubleSimpleAveraging, SGMTripleAveraging
from nsopy.methods.bundle import CuttingPlanesMethod, BundleMethod

AVAILABLE_METHODS = (
    'SG 1/k',
    'SG const',
    'UPGM',
    'UDGM',
    'UFGM',
    'DSA',
    'TA 1',
    'TA 2',
    'CP',
    'bundle'
)


def DualMethodsFactory(inner_problem, method, param=0):
    ###############
    # Subgradient #
    ###############
    if method == 'SG 1/k':
        if param == 0:
            return SubgradientMethod(oracle=inner_problem.oracle,
                                     projection_function=inner_problem.projection_function,
                                     dimension=inner_problem.dimension,
                                     stepsize_rule='1/k',
                                     sense='max')
        else:
            return SubgradientMethod(oracle=inner_problem.oracle,
                                     projection_function=inner_problem.projection_function,
                                     dimension=inner_problem.dimension,
                                     stepsize_rule='1/k',
                                     stepsize_0=param,
                                     sense='max')
    elif method == 'SG const':
        if param == 0:
            return SubgradientMethod(oracle=inner_problem.oracle,
                                     projection_function=inner_problem.projection_function,
                                     dimension=inner_problem.dimension,
                                     stepsize_rule='constant',
                                     sense='max')
        else:
            return SubgradientMethod(oracle=inner_problem.oracle,
                                     projection_function=inner_problem.projection_function,
                                     dimension=inner_problem.dimension,
                                     stepsize_rule='constant',
                                     stepsize_0=param,
                                     sense='max')
    #############
    # Universal #
    #############
    elif method == 'UPGM':
        if param == 0:
            from nsopy.methods.universal import UGM_DEFAULT_EPSILON
            epsilon = UGM_DEFAULT_EPSILON
        else:
            epsilon = param

        return UniversalPGM(oracle=inner_problem.oracle,
                            projection_function=inner_problem.projection_function,
                            dimension=inner_problem.dimension,
                            epsilon=epsilon,
                            sense='max')
    elif method == 'UDGM':
        if param == 0:
            from nsopy.methods.universal import UGM_DEFAULT_EPSILON
            epsilon = UGM_DEFAULT_EPSILON
        else:
            epsilon = param

        return UniversalDGM(oracle=inner_problem.oracle,
                            projection_function=inner_problem.projection_function,
                            dimension=inner_problem.dimension,
                            epsilon=epsilon,
                            sense='max')
    elif method == 'UFGM':
        if param == 0:
            from nsopy.methods.universal import UGM_DEFAULT_EPSILON
            epsilon = UGM_DEFAULT_EPSILON
        else:
            epsilon = param

        return UniversalFGM(oracle=inner_problem.oracle,
                            projection_function=inner_problem.projection_function,
                            dimension=inner_problem.dimension,
                            epsilon=epsilon,
                            sense='max')
    #####################
    # Quasi Monotone SG #
    #####################
    elif method == 'DSA':
        if param == 0:
            from nsopy.methods.quasi_monotone import METHOD_QUASI_MONOTONE_DEFAULT_GAMMA
            gamma = METHOD_QUASI_MONOTONE_DEFAULT_GAMMA
        else:
            gamma = param
        return SGMDoubleSimpleAveraging(oracle=inner_problem.oracle,
                                        projection_function=inner_problem.projection_function,
                                        dimension=inner_problem.dimension,
                                        gamma=gamma,
                                        sense='max')
    elif method == 'TA 1':
        if param == 0:
            from nsopy.methods.quasi_monotone import METHOD_QUASI_MONOTONE_DEFAULT_GAMMA
            gamma = METHOD_QUASI_MONOTONE_DEFAULT_GAMMA
        else:
            gamma = param
        return SGMTripleAveraging(oracle=inner_problem.oracle,
                                  projection_function=inner_problem.projection_function,
                                  dimension=inner_problem.dimension,
                                  variant=1,
                                  gamma=gamma,
                                  sense='max')
    elif method == 'TA 2':
        if param == 0:
            from nsopy.methods.quasi_monotone import METHOD_QUASI_MONOTONE_DEFAULT_GAMMA
            gamma = METHOD_QUASI_MONOTONE_DEFAULT_GAMMA
        else:
            gamma = param
        return SGMTripleAveraging(oracle=inner_problem.oracle,
                                  projection_function=inner_problem.projection_function,
                                  dimension=inner_problem.dimension,
                                  variant=2,
                                  gamma=gamma,
                                  sense='max')
    #########################
    # Cutting Planes/Bundle #
    #########################
    elif method == 'CP':
        print('Cutting Planes instantiated. Remember to call method.set_dual_domain().')
        if param == 0:
            from nsopy.methods.bundle import DEFAULT_EPSILON
            epsilon = DEFAULT_EPSILON
        else:
            epsilon = param
        return CuttingPlanesMethod(inner_problem.oracle,
                                   inner_problem.projection_function,
                                   dimension=inner_problem.dimension,
                                   epsilon=epsilon,
                                   sense='max')

    elif method == 'bundle':
        print('Bundle Method instantiated. Remember to call method.set_dual_domain().')
        if param == 0:
            from nsopy.methods.bundle import DEFAULT_EPSILON
            epsilon = DEFAULT_EPSILON
        else:
            epsilon = param

        # We use the default value for mu
        from nsopy.methods.bundle import DEFAULT_MU
        return BundleMethod(inner_problem.oracle,
                            inner_problem.projection_function,
                            dimension=inner_problem.dimension,
                            epsilon=epsilon,
                            mu=DEFAULT_MU,
                            sense='max')