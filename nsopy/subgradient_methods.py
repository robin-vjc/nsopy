# Classes incorporating the dual subgradient methods implemented in this package.
# New nsopy (and corresponding loggers) can be added by following the templates TemplateMethod, TemplateMethodLogger.

from __future__ import print_function
from __future__ import division
from nsopy.method_loggers import Observable
from nsopy.base import DualMethod
import numpy as np


#######################
# IMPLEMENTED METHODS #
#######################

class SubgradientMethod(DualMethod, Observable):
    """ Standard subgradient method """
    def __init__(self, oracle, projection_function, dimension=0, stepsize_rule='1/k', stepsize_0=1.0):
        super(SubgradientMethod, self).__init__()

        self.desc = 'SG, $s_0 = {}$'.format(stepsize_0)

        self.oracle = oracle
        self.projection_function = projection_function
        self.stepsize_rule = stepsize_rule

        self.iteration_number = 1
        self.oracle_calls = 0
        # self.stepsize_0 = stepsize_0 * np.ones(1, dtype=float)  # ensures it's float, for division
        self.stepsize_0 = float(stepsize_0)  # ensures it's float, for division

        self.d_k = np.zeros(1, dtype=float)
        if dimension == 0:
            self.lambda_k = self.projection_function(0)
            self.dimension = len(self.lambda_k)
        else:
            self.dimension = dimension
            self.lambda_k = self.projection_function(np.zeros(self.dimension, dtype=float))
        self.x_k = 0

        # for record keeping
        self.method_name = 'SG'
        self.parameter = stepsize_0

    # def dual_step(self, stepsize_rule='1/k'):
    def dual_step(self):
        # get subgradient
        self.x_k, self.d_k, diff_d_k = self.oracle(self.lambda_k)
        # log signal to any observers connected
        self.notify_observers()

        # print(diff_d_k)
        self.oracle_calls += 1

        if self.stepsize_rule == '1/k':
            self.desc = 'SG 1/k, $s_0 = {}$'.format(self.stepsize_0)
            self.method_name = 'SG 1/k'
            stepsize = self.stepsize_0 / self.iteration_number
        elif self.stepsize_rule == 'constant':
            self.desc = 'SG const,  $s_0 = {}$'.format(self.stepsize_0)
            self.method_name = 'SG const'
            stepsize = self.stepsize_0
        else:
            print('Warning: subgradient variant not recognized. Valid options: constant, 1/k.'
                  'Using constant stepsize.')
            stepsize = self.stepsize_0

        # perform dual step
        # lambda_kp1 = P_{lambda>=0} (lambda_k + stepsize*diff_d_k)
        # self.lambda_k += stepsize * diff_d_k  # this doesnt work anymore with the new version of numpy.
        self.lambda_k = self.lambda_k + stepsize * diff_d_k
        self.lambda_k = self.projection_function(self.lambda_k)

        self.iteration_number += 1

        # OLD
        # log signal to any observers connected
        # self.notify_observers()

