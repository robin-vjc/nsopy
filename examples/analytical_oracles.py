from __future__ import print_function

""" Small analytical models to test dual solvers. """
import numpy as np


class AnalyticalExampleInnerProblem(object):
    """
    This example has 2 inequalities that are dualized.
    Considers problem
        min     -0.5x1 - x2 + x3
        s.t.    0.5x1 + 0.5x2 + x3 >= 1
                x1 + x2 <= 1
                0 <= x1,x2,x3 <= 1
                x1 binary.
    and constructs associated dual function (and corresponding inner problem).
    The analytical solution for the dual is (optimizer is NOT unique!!):
        lambda_1* = 1
        lambda_2* = range[1, 1.5]
        d(lambda*) = -0.5
    """
    def __init__(self):
        self.n_constr = 2

    def oracle(self, lambda_k):
        if not type(lambda_k) == np.ndarray:
            print('WARNING: lambda_k should be a numpy array.')

        ###########################
        # Construct inner problem #
        ###########################

        # We could construct it as a Gurobi model
        ##########################################
        # m = gb.Model('inner_problem')
        # x = {}
        # x[1] = m.addVar(vtype=gb.GRB.BINARY, name='x1')
        # x[2] = m.addVar(lb=0, ub=1, vtype=gb.GRB.CONTINUOUS, name='x2')
        # x[3] = m.addVar(lb=0, ub=1, vtype=gb.GRB.CONTINUOUS, name='x3')
        # m.update()
        #
        # # Objective modified according to received 'prices'
        # m.setObjective((-0.5 - 0.5*lambda_k[0] + lambda_k[1])*x[1] +
        #                (-1 - 0.5*lambda_k[0] + lambda_k[1])*x[2] +
        #                (1 - lambda_k[0])*x[3] +
        #                lambda_k[0] - lambda_k[1],
        #                gb.GRB.MINIMIZE)
        # Solve
        #########
        # m.setParam('OutputFlag', False)
        # m.optimize()

        # But it's much easier to solve it analytically..
        ##################################################
        # Since the problem is trivial we have the analytical solution for the inner problem: we are
        # MININIMIZING in the inner problem. Feasible set is the cube. Hence, if the corresponding
        # objective term is negative, we set variable to 1, otherwise to 0.
        c = np.zeros(3, dtype=float)
        x_k = np.zeros(3, dtype=float)

        c[0] = - 0.5 - 0.5*lambda_k[0] + lambda_k[1]
        c[1] = - 1 - 0.5*lambda_k[0] + lambda_k[1]
        c[2] = + 1 - lambda_k[0]

        x_k[0] = 1 if c[0] < 0 else 0
        x_k[1] = 1 if c[1] < 0 else 0
        x_k[2] = 1 if c[2] < 0 else 0

        diff_d_k = np.zeros(2)
        diff_d_k[0] = 1 - 0.5*x_k[0] - 0.5*x_k[1] - x_k[2]
        diff_d_k[1] = x_k[0] + x_k[1] -1

        d_k = c[0]*x_k[0] + c[1]*x_k[1] + c[2] *x_k[2] + lambda_k[0] - lambda_k[1]

        return x_k, d_k, diff_d_k

    def projection_function(self, lambda_k):
        # simply project lambda_k on the positive orthant
        return np.maximum(lambda_k, 0)


class SecondAnalyticalExampleInnerProblem(object):
    """
    This example has 1 equality and 1 inequality.
    Considers problem
        min     -x1 - x2
        s.t.    x1 - x2 + 0.5 = 0
                x1 + x2 <= 1
                0 <= x1,x2 <= 1
    and constructs associated dual function (and corresponding inner problem)
    - "mu" (lambda_1) associated to equality
    - lambda_2 associated to inequality

    The analytical solution for the dual is:
        lambda_1* = 1
        mu* (lambda_2*) = 0
        d* = -1

    The dual function is:
    [A] (lambda >= 1 - mu), (lambda >= mu + 1)
    d(lambda, mu) = -lambda + 0.5mu
    [B] >=, <=
    d(lambda, mu) = - 1 - 0.5mu
    [C] <=, <=
    d(lambda, mu) = lambda + 0.5mu - 2
    [D] <=, >=
    d(lambda, mu) = -1 + 1.5mu
    """
    def __init__(self):
        self.n_constr = 2
        # below for record keeping
        self.instance_name = 'Second Analytical Example'
        self.instance_type = 'analytical'
        self.instance_subtype = 'analytical'

    def oracle(self, lambda_k):
        if not type(lambda_k) == np.ndarray:
            print('WARNING: lambda_k should be a numpy array.')

        c = np.zeros(2, dtype=float)
        x_k = np.zeros(2, dtype=float)

        c[0] = - 1 + lambda_k[0] + lambda_k[1]
        c[1] = - 1 + lambda_k[0] - lambda_k[1]

        x_k[0] = 1 if c[0] < 0 else 0
        x_k[1] = 1 if c[1] < 0 else 0

        diff_d_k = np.zeros(2)
        diff_d_k[0] = x_k[0] + x_k[1] - 1
        diff_d_k[1] = x_k[0] - x_k[1] + 0.5

        d_k = c[0]*x_k[0] + c[1]*x_k[1] - lambda_k[0] + 0.5*lambda_k[1]

        return x_k, d_k, diff_d_k

    def projection_function(self, lambda_k):
        # simply project lambda_k[0] on the positive orthant; lambda_k[1] free
        return np.array([np.maximum(lambda_k[0], 0), lambda_k[1]])


class ConstrainedDualAnalyticalExampleInnerProblem(object):
    """
    This example tests whether methods can cope correctly with constrained dual sets.

    Consider problem
        min     -x1 + x2 - 0.5y
        s.t.    y >= -x1 + 1
                y >= x2
                0 <= x1,x2,y <= 1
    which we rewrite as
        min     -x1 + x2 - 0.5y
        s.t.    y1 = y
                y2 = y
                y1 >= -x1 + 1
                y2 >= x2
                0 <= x1,x2,y1,y2 <= 1
    (note while y1 and y2 are in unit cube, y unbounded)
    Dual is then:
    d(lmd) = min_{x1,x2,y} -x1 + x2 - 0.5y + lmb1(y1-y) + lmb2(y2-y)
             s.t.   y1 >= -x1 + 1
                    y2 >= x2
                    0 <= x1,x2,y1,y2 <= 1
    reorganized as
    d(lmd) = min_{x1,x2,y} x1*(-1) + x2*(1) + y1*lmb1 + y2*lmb2 + y(0.5 - lmb1- lmb2)
             s.t.   y1 >= -x1 + 1
                    y2 >= x2
                    0 <= x1,x2,y1,y2 <= 1
    which again separates in the problems
    d(lmd) = min_{x1,y1} -x1 + y1*lmb1 +  min_{x2,y2} x2 + y2*lmb2 + min_y y(0.5 - lmb1- lmb2)
             s.t. y1 >= -x1 + 1           s.t. y2 >= x2              s.t.  NOTHING

    Notes:
    - since y is unbounded, unless lmb1+lmb2=0.5, the dual function is -infty (so could e.g. throw error then
    "oracle queried at point where d is not defined").

    Inner solutions:
    [1] x1* = 1
        y1* = { 1 if lmb1 <= 0
              { 0 if lmb1 > 0
    [2] x2* = 0
        y2* = { 1 if lmb2 <= 0
              { 0 if lmb1 > 0
    [3] if lmb1+lmb2 = 0.5, y*=0.
    else throw warning (and return d = -infty, diff_d_k = infty)
    """
    def __init__(self):
        self.n_constr = 2

    def oracle(self, lambda_k):
        if not type(lambda_k) == np.ndarray:
            print('WARNING: lambda_k should be a numpy array.')

        c = np.zeros(5, dtype=float)
        x_k = np.zeros(5, dtype=float)  # x1, x2, y1, y2, y

        c[0] = - 1  # x1
        c[1] = + 1  # x2
        c[2] = lambda_k[0]  # y1
        c[3] = lambda_k[1]  # y2
        c[4] = 0.5 - lambda_k[0] - lambda_k[1]

        x_k[0] = 1
        x_k[1] = 0
        x_k[2] = 1 if c[2] < 0 else 0
        x_k[3] = 1 if c[3] < 0 else 0
        x_k[4] = float(x_k[2] + x_k[3])/float(2)

        diff_d_k = np.zeros(2)
        diff_d_k[0] = x_k[2] - x_k[4]
        diff_d_k[1] = x_k[3] - x_k[4]

        if not abs(c[4] - 0) <= 0.01:
            print('querying the dual function at an undefined point')
            return x_k, -np.infty, np.array([np.infty, np.infty])

        d_k = c[0]*x_k[0] + c[1]*x_k[1] + c[2]*x_k[2] + c[3]*x_k[3] + c[4]*x_k[4]

        return x_k, d_k, diff_d_k

    def projection_function(self, lambda_k):
        # simply project lambda_k[0] on the positive orthant; lambda_k[1] free
        return np.array([(lambda_k[0]-lambda_k[1])/float(2) + 0.25, (lambda_k[1]-lambda_k[0])/float(2) + 0.25])


class BertsekasCounterExample(object):
    """
    This example is from
    - Dimitri Bertsekas, Convex Optimization Algorithms, Athena Scientific Belmont, 2015
    and considers the problem:

    """
    def __init__(self):
        self.n_constr = 2

    def oracle(self, lambda_k):
        if not type(lambda_k) == np.ndarray:
            print('WARNING: lambda_k should be a numpy array.')

        c = np.zeros(5, dtype=float)
        x_k = np.zeros(5, dtype=float)  # x1, x2, y1, y2, y

        c[0] = - 1  # x1
        c[1] = + 1  # x2
        c[2] = lambda_k[0]  # y1
        c[3] = lambda_k[1]  # y2
        c[4] = 0.5 - lambda_k[0] - lambda_k[1]

        x_k[0] = 1
        x_k[1] = 0
        x_k[2] = 1 if c[2] < 0 else 0
        x_k[3] = 1 if c[3] < 0 else 0
        x_k[4] = float(x_k[2] + x_k[3])/float(2)

        diff_d_k = np.zeros(2)
        diff_d_k[0] = x_k[2] - x_k[4]
        diff_d_k[1] = x_k[3] - x_k[4]

        if not abs(c[4] - 0) <= 0.01:
            print('querying the dual function at an undefined point')
            return x_k, -np.infty, np.array([np.infty, np.infty])

        d_k = c[0]*x_k[0] + c[1]*x_k[1] + c[2]*x_k[2] + c[3]*x_k[3] + c[4]*x_k[4]

        return x_k, d_k, diff_d_k

    def projection_function(self, lambda_k):
        # simply project lambda_k[0] on the positive orthant; lambda_k[1] free
        return np.array([(lambda_k[0]-lambda_k[1])/float(2) + 0.25, (lambda_k[1]-lambda_k[0])/float(2) + 0.25])
