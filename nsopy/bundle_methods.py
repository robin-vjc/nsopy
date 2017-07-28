##################################
# CUTTING PLANE & BUNDLE METHODS #
##################################
# Implementation based on
# [1] Alexandre Belloni, Lecture Notes for IAP 2005 Course Introduction to Bundle Methods.
# pdf originally at: https://faculty.fuqua.duke.edu/~abn5/LecturesIntroBundle.pdf
# copy of the pdf in "./nsopy/doc/"

from __future__ import print_function
from __future__ import division
from nsopy.method_loggers import Observable
from nsopy.base import DualMethod
import numpy as np
import copy
try:
    import gurobipy as gb
except ImportError:
    print('Gurobi required for Cutting Planes and Bundle nsopy.')


DEFAULT_EPSILON = 0.01
DEFAULT_MU = 0.5
SEARCH_BOX_MIN = -10
SEARCH_BOX_MAX = 10


class CuttingPlanesMethod(DualMethod, Observable):
    """
    Implementation of Algorithm (CP) in [1], p.19.
    """

    def __init__(self, oracle, projection_function, dimension=0, epsilon=DEFAULT_EPSILON):
        super(CuttingPlanesMethod, self).__init__()
        self.desc = 'Cutting Planes, $\epsilon = {}$'.format(epsilon)

        self.oracle = oracle
        self.projection_function = projection_function

        self.iteration_number = 1
        self.oracle_calls = 0
        self.epsilon = epsilon
        self.optimizer_not_yet_found = 1

        if dimension == 0:
            self.lambda_k = self.projection_function(0)
            self.dimension = len(self.lambda_k)
        else:
            self.dimension = dimension
            self.lambda_k = self.projection_function(np.zeros(self.dimension, dtype=float))

        self.d_k = np.zeros(1, dtype=float)
        self.x_k = 0
        self.diff_d_k = 0

        # Step 1
        # search box C
        self.lambda_min = SEARCH_BOX_MIN
        self.lambda_max = SEARCH_BOX_MAX
        # initial function value
        self.f_hat_lambda_k = -np.infty
        self.bundle = []


        # --------------------------------------------------- #
        # Initialize LP model of cutting plane
        self.bundle_model = gb.Model()
        # self.bundle_model.setParam(u'MIPGap', 0.745)
        self.bundle_model.setParam(u'TimeLimit', 360)
        self.bundle_model.setParam('OutputFlag', False)
        self.r = self.bundle_model.addVar(obj=1, lb=-gb.GRB.INFINITY)
        self.lmd = {}
        for i in range(self.dimension):
            self.lmd[i] = self.bundle_model.addVar(vtype=gb.GRB.CONTINUOUS,
                                                   obj=0,
                                                   lb=self.lambda_min,
                                                   ub=self.lambda_max)
        self.constraints = {}
        self.bundle_model.update()
        # --------------------------------------------------- #

        # for record keeping
        self.method_name = 'CP'
        self.parameter = epsilon

    def dual_step(self):
        if self.optimizer_not_yet_found:
            # Step 2
            # print('before oracle call')
            self.x_k, self.d_k, self.diff_d_k = self.oracle(self.lambda_k)
            # print('after oracle call')
            self.oracle_calls += 1

            # Step 3
            delta_k = - self.d_k - self.f_hat_lambda_k

            # print("########### step k={} ############".format(self.iteration_number))
            # print("delta_k = "+str(delta_k))
            # assert delta_k >= 0

            # Step 4
            if delta_k < self.epsilon:
                # print('we have found a point satisfying optimality gap')
                # optimizer found
                self.optimizer_not_yet_found = 0
            else:
                # Step 5
                a = - self.diff_d_k
                b = - self.d_k - np.dot(-self.diff_d_k, self.lambda_k)
                self.bundle.append((a, b))  # f_hat(lambda) = a*lambda + b
                # print("bundle model: "+str(self.bundle))
                # Step 6, compute and solve LP
                self.f_hat_lambda_k, self.lambda_k = self.min_of_bundle()

        # Step 7
        self.iteration_number += 1

        # log signal to any observers connected
        self.notify_observers()

    def min_of_bundle(self):
        # add new constraint
        a, b = self.bundle[-1]
        # print(a,b)
        # self.bundle_model.addConstr(self.r >= gb.quicksum([a[i]*self.lmd[i] for i in range(self.n_constr)]) + b)
        self.constraints[self.iteration_number] = self.bundle_model.addConstr(
            self.r >= gb.quicksum([a[i] * self.lmd[i] for i in range(self.dimension)]) + b)

        self.bundle_model.update()
        # print("constr coeffs lambda mu: " + str(
        #     [self.bundle_model.getCoeff(self.constraints[self.iteration_number], self.lmd[i]) for i in
        #      range(self.n_constr)]))
        # print("constr coeffs r: " + str(self.bundle_model.getCoeff(self.constraints[self.iteration_number], self.r)))

        # self.bundle_model.write('test_model.lp')
        self.bundle_model.optimize()
        optimizer = np.array([self.lmd[i].X for i in range(self.dimension)])
        # return 0, np.array([0, 0])
        # print("optimizer: " + str(optimizer))
        # print("val: " + str(self.r.X))

        return self.bundle_model.ObjVal, optimizer

    def set_dual_domain(self, type='free', param=0):
        # constrain dual domain
        if type == 'free':
            pass
        elif type == 'positive orthant':
            for i in range(self.dimension):
                self.bundle_model.addConstr(self.lmd[i] >= 0)
        elif type == 'sum to param':
            self.bundle_model.addConstr(gb.quicksum([self.lmd[i] for i in self.lmd]) == param)
        elif type == '2 stage smps':
            # custom made for this inner problem type
            inner_problem = self.oracle.im_self  # this feels dirty
            for i in range(inner_problem.n_x):
                self.bundle_model.addConstr(
                    gb.quicksum([self.lmd[i+sc*(inner_problem.n_x)] for sc in range(inner_problem.n_scenarios)]) == 0
                )
        elif type == 'mrf':
            for i in range(int(self.dimension/2)):
                self.bundle_model.addConstr(
                    self.lmd[i] + self.lmd[i + int(self.dimension / 2)] == 0
                )
        else:
            ValueError('Type of dual domain not recognized.')

        self.bundle_model.update()


class BundleMethod(DualMethod, Observable):
    """
    Implementation of Bundle Method, based on my paper, as adapted from algorithm (BA) in [1],
    p.21 and Algorithm 7.3 in [2], p.374 (for the constrained dual case).
    """

    def __init__(self, oracle, projection_function, dimension=0, epsilon=DEFAULT_EPSILON, mu=DEFAULT_MU):
        super(BundleMethod, self).__init__()
        self.desc = 'Bundle Method, $\epsilon = {}, \mu = {}$'.format(epsilon, mu)

        self.oracle = oracle
        self.projection_function = projection_function

        self.iteration_number = 1
        self.oracle_calls = 0
        self.optimizer_not_yet_found = 1

        # Initialization.
        self.epsilon = epsilon
        self.mu = mu
        self.gamma = 0.5

        # current iterate values
        if dimension == 0:
            self.lambda_k = self.projection_function(0)
            self.dimension = len(self.lambda_k)
        else:
            self.dimension = dimension
            self.lambda_k = self.projection_function(np.zeros(self.dimension, dtype=float))

        self.x_k = 0
        self.d_k = 0
        self.diff_d_k = 0

        # "hat" values
        self.lambda_hat_k = 0
        self.d_hat_k = 0
        self.diff_d_hat_k = 0

        # search box C
        # self.lambda_min = SEARCH_BOX_MIN
        # self.lambda_max = SEARCH_BOX_MAX
        # initial function value
        self.f_hat_lambda_k = -np.infty

        # bundle model
        self.bundle = []

        # --------------------------------------------------- #
        # Initialize LP model of cutting plane
        self.bundle_model = gb.Model()
        # self.bundle_model.setParam(u'MIPGap', 0.745)
        # self.bundle_model.setParam(u'TimeLimit', 120)
        self.bundle_model.setParam('OutputFlag', False)
        self.r = self.bundle_model.addVar(obj=1, lb=-gb.GRB.INFINITY)
        self.lmd = {}
        for i in range(self.dimension):
            self.lmd[i] = self.bundle_model.addVar(vtype=gb.GRB.CONTINUOUS,
                                                   obj=0,
                                                   lb=-gb.GRB.INFINITY,
                                                   )
        self.constraints = {}
        self.bundle_model.update()
        # --------------------------------------------------- #

        # for record keeping
        self.method_name = 'bundle'
        self.parameter = epsilon

    def dual_step(self):
        if self.iteration_number == 1:
            self.x_k, self.d_k, self.diff_d_k = self.oracle(self.lambda_k)
            self.oracle_calls += 1

            # "hat" values
            self.lambda_hat_k = self.projection_function(np.zeros(self.dimension, dtype=float))
            self.d_hat_k = copy.deepcopy(self.d_k)
            self.diff_d_hat_k = copy.deepcopy(self.diff_d_k)

            a = - self.diff_d_k
            b = - self.d_k - np.dot(-self.diff_d_k, self.lambda_k)
            self.bundle.append((a, b))  # f_hat(lambda) = a*lambda + b

        if self.optimizer_not_yet_found:
            # Step 1
            d_star_k, self.lambda_k = self.min_of_bundle()

            delta_k = abs(-self.d_k -d_star_k)

            # Step 2, 3
            if delta_k < self.epsilon:
                self.optimizer_not_yet_found = 0
                # print('optimizer found')
            else:
                # Step 4
                self.x_k, self.d_k, self.diff_d_k = self.oracle(self.lambda_k)
                self.oracle_calls += 1

                a = - self.diff_d_k
                b = - self.d_k - np.dot(-self.diff_d_k, self.lambda_k)
                self.bundle.append((a, b))  # f_hat(lambda) = a*lambda + b

                # Step 5
                if self.d_k - self.d_hat_k >= self.gamma*self.epsilon:
                    # SERIOUS STEP
                    self.d_hat_k = self.d_k
                    self.lambda_hat_k = self.lambda_k

                # NULL STEP: no change
                # else:
                #   self.d_hat_k = self.d_hat_k
                #   self.lambda_hat_k = self.lambda_hat_k

        # Step 6
        self.iteration_number += 1
        # log signal to any observers connected
        self.notify_observers()

    def min_of_bundle(self):
        # set objective
        objective = self.r + float(self.mu)/float(2)*gb.quicksum(
            [(self.lmd[i] - self.lambda_hat_k[i]) * (self.lmd[i] - self.lambda_hat_k[i]) for i in range(self.dimension)]
        )
        self.bundle_model.setObjective(objective)

        # rest below is same as cutting planes
        # add new constraint
        a, b = self.bundle[-1]
        # print(a,b)
        # self.bundle_model.addConstr(self.r >= gb.quicksum([a[i]*self.lmd[i] for i in range(self.n_constr)]) + b)
        self.constraints[self.iteration_number] = self.bundle_model.addConstr(
            self.r >= gb.quicksum([a[i] * self.lmd[i] for i in range(self.dimension)]) + b)

        self.bundle_model.update()
        # print("constr coeffs lambda mu: " + str(
        #     [self.bundle_model.getCoeff(self.constraints[self.iteration_number], self.lmd[i]) for i in
        #      range(self.n_constr)]))
        # print("constr coeffs r: " + str(self.bundle_model.getCoeff(self.constraints[self.iteration_number], self.r)))

        # self.bundle_model.write('test_model.lp')
        self.bundle_model.optimize()
        optimizer = np.array([self.lmd[i].X for i in range(self.dimension)])
        # return 0, np.array([0, 0])
        # print("optimizer: " + str(optimizer))
        # print("val: " + str(self.r.X))

        return self.bundle_model.ObjVal, optimizer

    def set_dual_domain(self, type='free', param=0):
        # constrain dual domain
        if type == 'free':
            pass
        elif type == 'positive orthant':
            for i in range(self.dimension):
                self.bundle_model.addConstr(self.lmd[i] >= 0)
        elif type == 'sum to param':
            self.bundle_model.addConstr(gb.quicksum([self.lmd[i] for i in self.lmd]) == param)
        elif type == '2 stage smps':
            # custom made for this inner problem type
            inner_problem = self.oracle.im_self  # this feels dirty
            for i in range(inner_problem.n_x):
                self.bundle_model.addConstr(
                    gb.quicksum([self.lmd[i+sc*(inner_problem.n_x)] for sc in range(inner_problem.n_scenarios)]) == 0
                )
        elif type == 'mrf':
            for i in range(int(self.dimension/2)):
                self.bundle_model.addConstr(
                    self.lmd[i] + self.lmd[i + int(self.dimension / 2)] == 0
                )
        else:
            ValueError('Type of dual domain not recognized.')

        self.bundle_model.update()

