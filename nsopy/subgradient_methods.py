# Classes incorporating the dual subgradient methods implemented in this package.
# New nsopy (and corresponding loggers) can be added by following the templates TemplateMethod, TemplateMethodLogger.

from __future__ import print_function
from __future__ import division
from nsopy.method_loggers import Observable
from nsopy.base import SolutionMethod
import numpy as np


#######################
# IMPLEMENTED METHODS #
#######################
from nsopy.utils import invert_oracle_sense


class SubgradientMethod(SolutionMethod, Observable):
    """ Standard subgradient method """
    def __init__(self, oracle, projection_function, dimension=0, stepsize_rule='1/k', stepsize_0=1.0, sense='min'):
        super(SubgradientMethod, self).__init__()

        self.desc = 'SG, $s_0 = {}$'.format(stepsize_0)

        if sense == 'min':
            self.oracle = invert_oracle_sense(oracle)  # all methods have been coded to maximize the oracle model
        elif sense == 'max':
            self.oracle = oracle
        else:
            raise ValueError('Sense should be either "min" or "max"')

        self.projection_function = projection_function

        self.stepsize_0 = float(stepsize_0)  # ensures it's float, for division
        assert stepsize_rule in ['1/k', 'constant', '1/sqrt(k)'], "stepsize_rule has to be either '1/k', 'constant' ot '1/sqrt(k)'"
        self.stepsize_rule = stepsize_rule
        if self.stepsize_rule == '1/k':
            self.desc = 'SG 1/k, $s_0 = {}$'.format(self.stepsize_0)
            self.method_name = 'SG 1/k'
        elif self.stepsize_rule == 'constant':
            self.desc = 'SG const,  $s_0 = {}$'.format(self.stepsize_0)
            self.method_name = 'SG const'
        elif self.step() == '1/sqrt(k)':
            self.desc = 'SG 1/sqrt(k),  $s_0 = {}$'.format(self.stepsize_0)

        self.iteration_number = 1
        self.oracle_calls = 0

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

    def dual_step(self):
        # get subgradient
        self.x_k, self.d_k, diff_d_k = self.oracle(self.lambda_k)
        # log signal to any observers connected
        self.notify_observers()

        # print(diff_d_k)
        self.oracle_calls += 1

        if self.stepsize_rule == '1/k':
            stepsize = self.stepsize_0 / self.iteration_number
        elif self.stepsize_rule == 'constant':
            stepsize = self.stepsize_0
        elif self.step() == '1/sqrt(k)':
            stepsize = self.stepsize_0 / np.sqrt(self.iteration_number)

        # perform dual step
        # lambda_kp1 = P_{lambda>=0} (lambda_k + stepsize*diff_d_k)
        # self.lambda_k += stepsize * diff_d_k  # this doesnt work anymore with the new version of numpy.
        self.lambda_k = self.lambda_k + stepsize * diff_d_k
        self.lambda_k = self.projection_function(self.lambda_k)

        self.iteration_number += 1
