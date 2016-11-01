# Implementation of the methods in
# Yu. Nesterov , V. Shikhman, "Quasi-monotone Subgradient Methods for Nonsmooth Convex Minimization"
# Journal of Optimization Theory and Applications
# http://link.springer.com/article/10.1007/s10957-014-0677-5

from __future__ import division
from method_loggers import Observable
from base import DualMethod
import numpy as np

METHOD_QUASI_MONOTONE_DEFAULT_GAMMA = 1.0


# Implementation of "Subgradient Method with Double Simple Averaging", p.928.
class SGMDoubleSimpleAveraging(DualMethod, Observable):
    """ Implementation of a dual method """
    def __init__(self, oracle, projection_function, n_constr=0, gamma=METHOD_QUASI_MONOTONE_DEFAULT_GAMMA):
        super(SGMDoubleSimpleAveraging, self).__init__()

        self.desc = 'DSA, $\gamma = {}$'.format(gamma)
        self.oracle = oracle
        self.projection_function = projection_function

        self.oracle_calls = 0
        self.n_constr = n_constr
        self.iteration_number = 0

        self.d_k = 0
        self.lambda_k = np.zeros(self.n_constr, dtype=float)
        self.x_k = 0

        self.gamma = gamma
        self.s_k = np.zeros(self.n_constr, dtype=float)  # this stores \sum_{k=0}^t diff_d_k

        # for record keeping
        self.method_name = 'DSA'
        self.parameter = gamma

    def dual_step(self):
        self.x_k, self.d_k, self.diff_d_k = self.oracle(self.lambda_k)
        self.oracle_calls += 1

        self.s_k += self.diff_d_k
        lambda_k_plus = float(1.0)/float(self.gamma*np.sqrt(self.iteration_number+1)) * self.s_k
        lambda_k_plus = self.projection_function(lambda_k_plus)

        self.lambda_k = float(self.iteration_number+1)/float(self.iteration_number+2)*self.lambda_k \
                        + float(1.0)/float(self.iteration_number+2)*lambda_k_plus

        self.iteration_number += 1
        self.notify_observers()


# Implementation of "Subgradient Method with Triple Averaging", p.930.
class SGMTripleAveraging(DualMethod, Observable):
    """ Implementation of a dual method """
    def __init__(self, oracle, projection_function, n_constr=0, variant=1, gamma=METHOD_QUASI_MONOTONE_DEFAULT_GAMMA):
        super(SGMTripleAveraging, self).__init__()

        self.desc = 'TA, $\gamma = {}$'.format(gamma)

        self.oracle = oracle
        self.projection_function = projection_function

        self.oracle_calls = 0
        self.n_constr = n_constr
        self.iteration_number = 0
        self.gamma = gamma
        self.variant = variant

        self.d_k = 0
        self.lambda_k = np.zeros(self.n_constr, dtype=float)
        self.lambda_0 = np.zeros(self.n_constr, dtype=float)
        self.x_k = 0

        self.s_k = np.zeros(self.n_constr, dtype=float)  # this stores \sum_{k=0}^t diff_d_k

        # for record keeping
        self.method_name = 'TA'
        self.parameter = gamma

    def dual_step(self):
        self.x_k, self.d_k, self.diff_d_k = self.oracle(self.lambda_k)
        self.oracle_calls += 1

        # step 1
        if self.variant == 1:
            self.desc = 'TA 1, $\gamma = {}$'.format(self.gamma)
            self.method_name = 'TA 1'
            self.s_k += self.diff_d_k
            gamma_t = self.gamma*np.sqrt(self.iteration_number+1)
            gamma_t_plus_1 = self.gamma*np.sqrt(self.iteration_number+2)
            tau_t = float(1)/float(self.iteration_number+2)

        elif self.variant == 2:
            self.desc = 'TA 2, $\gamma = {}$'.format(self.gamma)
            self.method_name = 'TA 2'
            self.s_k += (self.iteration_number+1)*self.diff_d_k
            # OLD Version: verbatim as in Paper
            # gamma_t = (self.iteration_number+1)**(float(3.0)/float(2.0))
            # gamma_t_plus_1 = (self.iteration_number+2)**(float(3.0)/float(2.0))
            # NEW Version: with tuning gamma
            gamma_t = self.gamma*(self.iteration_number+1)**(float(3.0)/float(2.0))
            gamma_t_plus_1 = self.gamma*(self.iteration_number+2)**(float(3.0)/float(2.0))
            tau_t = float(self.iteration_number+1)/float(sum([i for i in range(self.iteration_number+2)]))

        else:
            raise ValueError('Supported variants are 1: a_t = 1, gamma_t = gamma*sqrt(t+1) and '
                             '2: a_t = t, gamma_t = t^(3/2).')

        lambda_k_plus = float(1.0)/float(gamma_t) * self.s_k
        lambda_k_plus = self.projection_function(lambda_k_plus)

        # step 2
        lambda_k_hat = float(gamma_t)/float(gamma_t_plus_1) * lambda_k_plus \
                       + (1 - float(gamma_t)/float(gamma_t_plus_1))* self.lambda_0


        # step 4
        self.lambda_k = (1-tau_t)*self.lambda_k + tau_t*lambda_k_hat

        self.iteration_number += 1
        self.notify_observers()