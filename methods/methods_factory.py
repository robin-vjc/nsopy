# Centralize instantiation of dual methods. Useful in particular since the different
# method classes have slightly different instantiation parameters.

from methods.subgradient_methods import SubgradientMethod
from methods.universal_gradient_methods import UniversalPGM, UniversalDGM, UniversalFGM
from methods.quasi_monotone_subgradient_methods import SGMDoubleSimpleAveraging, SGMTripleAveraging
from methods.bundle_methods import CuttingPlanesMethod, BundleMethod

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
                                     n_constr=inner_problem.n_constr,
                                     stepsize_rule='1/k')
        else:
            return SubgradientMethod(oracle=inner_problem.oracle,
                                     projection_function=inner_problem.projection_function,
                                     n_constr=inner_problem.n_constr,
                                     stepsize_rule='1/k',
                                     stepsize_0=param)
    elif method == 'SG const':
        if param == 0:
            return SubgradientMethod(oracle=inner_problem.oracle,
                                     projection_function=inner_problem.projection_function,
                                     n_constr=inner_problem.n_constr,
                                     stepsize_rule='constant')
        else:
            return SubgradientMethod(oracle=inner_problem.oracle,
                                     projection_function=inner_problem.projection_function,
                                     n_constr=inner_problem.n_constr,
                                     stepsize_rule='constant',
                                     stepsize_0=param)
    #############
    # Universal #
    #############
    elif method == 'UPGM':
        if param == 0:
            from methods.universal_gradient_methods import METHOD_UNIVERSAL_GRADIENT_DEFAULT_EPSILON
            epsilon = METHOD_UNIVERSAL_GRADIENT_DEFAULT_EPSILON
        else:
            epsilon = param

        return UniversalPGM(oracle=inner_problem.oracle,
                            projection_function=inner_problem.projection_function,
                            n_constr=inner_problem.n_constr,
                            epsilon=epsilon)
    elif method == 'UDGM':
        if param == 0:
            from methods.universal_gradient_methods import METHOD_UNIVERSAL_GRADIENT_DEFAULT_EPSILON
            epsilon = METHOD_UNIVERSAL_GRADIENT_DEFAULT_EPSILON
        else:
            epsilon = param

        return UniversalDGM(oracle=inner_problem.oracle,
                            projection_function=inner_problem.projection_function,
                            n_constr=inner_problem.n_constr,
                            epsilon=epsilon)
    elif method == 'UFGM':
        if param == 0:
            from methods.universal_gradient_methods import METHOD_UNIVERSAL_GRADIENT_DEFAULT_EPSILON
            epsilon = METHOD_UNIVERSAL_GRADIENT_DEFAULT_EPSILON
        else:
            epsilon = param

        return UniversalFGM(oracle=inner_problem.oracle,
                            projection_function=inner_problem.projection_function,
                            n_constr=inner_problem.n_constr,
                            epsilon=epsilon)
    #####################
    # Quasi Monotone SG #
    #####################
    elif method == 'DSA':
        if param == 0:
            from methods.quasi_monotone_subgradient_methods import METHOD_QUASI_MONOTONE_DEFAULT_GAMMA
            gamma = METHOD_QUASI_MONOTONE_DEFAULT_GAMMA
        else:
            gamma = param
        return SGMDoubleSimpleAveraging(oracle=inner_problem.oracle,
                                        projection_function=inner_problem.projection_function,
                                        n_constr=inner_problem.n_constr,
                                        gamma=gamma)
    elif method == 'TA 1':
        if param == 0:
            from methods.quasi_monotone_subgradient_methods import METHOD_QUASI_MONOTONE_DEFAULT_GAMMA
            gamma = METHOD_QUASI_MONOTONE_DEFAULT_GAMMA
        else:
            gamma = param
        return SGMTripleAveraging(oracle=inner_problem.oracle,
                                  projection_function=inner_problem.projection_function,
                                  n_constr=inner_problem.n_constr,
                                  variant=1,
                                  gamma=gamma)
    elif method == 'TA 2':
        if param == 0:
            from methods.quasi_monotone_subgradient_methods import METHOD_QUASI_MONOTONE_DEFAULT_GAMMA
            gamma = METHOD_QUASI_MONOTONE_DEFAULT_GAMMA
        else:
            gamma = param
        return SGMTripleAveraging(oracle=inner_problem.oracle,
                                  projection_function=inner_problem.projection_function,
                                  n_constr=inner_problem.n_constr,
                                  variant=2,
                                  gamma=gamma)
    #########################
    # Cutting Planes/Bundle #
    #########################
    elif method == 'CP':
        print('Cutting Planes instantiated. Remember to call method.set_dual_domain().')
        if param == 0:
            from methods.bundle_methods import DEFAULT_EPSILON
            epsilon = DEFAULT_EPSILON
        else:
            epsilon = param
        return CuttingPlanesMethod(inner_problem.oracle,
                                   inner_problem.projection_function,
                                   n_constr=inner_problem.n_constr,
                                   epsilon=epsilon)

    elif method == 'bundle':
        print('Bundle Method instantiated. Remember to call method.set_dual_domain().')
        if param == 0:
            from methods.bundle_methods import DEFAULT_EPSILON
            epsilon = DEFAULT_EPSILON
        else:
            epsilon = param

        # We use the default value for mu
        from methods.bundle_methods import DEFAULT_MU
        return BundleMethod(inner_problem.oracle,
                            inner_problem.projection_function,
                            n_constr=inner_problem.n_constr,
                            epsilon=epsilon,
                            mu=DEFAULT_MU)