##############################
# UNIVERSAL GRADIENT METHODS #
##############################
# References
# [1] Universal Gradient Methods for Convex Optimization Problems, Yu. Nesterov, CORE Discussion Paper, 2013.
# Note: zeta(x,y) = ||y-x||^2_2 is used as the prox function, throughout.

from __future__ import division
from nsopy.method_loggers import Observable
from nsopy.base import DualMethod
import numpy as np
import copy

UGM_DEFAULT_EPSILON = 1.0
UGM_DEFAULT_L_0 = 1.1


def _bregman_map(M, lambda_k, diff_d_k):
    # Bregman map, according to [1], Eq. 2.9, with f(x) := -d(lambda), and M*psi(x,y) := M/2*||lambda-lambda_k||_2
    # Old:
    # lambda_bregman = lambda_k + float(1.0)/M*diff_d_k
    # return self._project_on_dual_feasible_set(lambda_bregman)
    return lambda_k + float(1.0)/M*diff_d_k


####################################
# UNIVERSAL PRIMAL GRADIENT METHOD #
####################################

class UniversalPGM(DualMethod, Observable):
    """
    Implementation of Algorithm (2.16) in [1], the Universal Primal Gradient Method.
    Note that the algorithm is written for the maximization of a convex function, while in the duality
    framework we maximize a concave fct. Hence, f(x) := -d(lambda)
    """
    def __init__(self, oracle, projection_function, dimension=0, epsilon=UGM_DEFAULT_EPSILON, averaging=False):
        """
        Averaging: Nesterov's nsopy give guarantees on variables marked with a tilde. Those are supposed to be the
        actual outputs of the method, but they require extra computations (evaluation of d(lambda_tilde)), and these can
        be carried out after the method has gone through all the required iterations. We therefore introduce the
        following additional attributed: lambda_hat_k and d_hat_k. These replace lambda_k and d_k.
        - averaging=True establishes that lambda_k and d_k (the method's outputs) are the lambda_tilde_k and d_tilde_k
        (i.e., those extra computations are carried out during execution, even thoug it wouldnt be in principle
        necessary)
        - averaging=False, method's outputs lambda_k and d_k

        TL;DR: both options deliver valid iterates, but only the iterates produced with averaging=True are endowed with
        his theoretical properties; on the other hand, they require extra computations (one extra oracle calls per
        iteration).
        """
        super(UniversalPGM, self).__init__()

        self.desc = 'UPGM, $\epsilon = {}$'.format(epsilon)
        self.oracle = oracle
        self.projection_function = projection_function

        self.iteration_number = 1
        self.oracle_calls = 0

        self.d_hat_k = np.zeros(1, dtype=float)
        if dimension == 0:
            self.lambda_hat_k = self.projection_function(0)
            self.dimension = len(self.lambda_hat_k)
        else:
            self.dimension = dimension
            self.lambda_hat_k = self.projection_function(np.zeros(self.dimension, dtype=float))
        self.x_hat_k = 0

        # specific to U-PGM
        self.diff_d_hat_k = 0
        self.L_k = float(UGM_DEFAULT_L_0)  # if you use something else, make sure it's a float!
        self.epsilon = float(epsilon)
        self.i_k = 0

        # -- Averaging -- Synthesize averaged outputs
        # Variables to synthesize solution from algorithm's process
        # records of d_tilda_k and lambda_tilda_k ("averages") according to Eqns. below 2.17
        self.S_k = float(1)/float(self.L_k)
        self.lambda_tilde_k = copy.deepcopy(self.lambda_hat_k)
        self.sum_lambda_tilde_k = copy.deepcopy(self.lambda_hat_k)  # \sum_i=0^k lambda_tilda_k
        self.d_tilde_k = 0
        self.sum_d_tilde_k = 0

        self.averaging = averaging
        if self.averaging:
            self.lambda_k = self.lambda_tilde_k
            self.d_k = self.d_tilde_k
            self.diff_d_k = self.diff_d_hat_k
            self.x_k = self.x_hat_k
        else:
            self.lambda_k = self.lambda_hat_k
            self.d_k = self.d_hat_k
            self.diff_d_k = self.diff_d_hat_k
            self.x_k = self.x_hat_k
        # -- Averaging --

        # for record keeping
        self.method_name = 'UPGM'
        self.parameter = epsilon

    def _bregman_map(self, M, lambda_k, subgrad_lambda_k):
        return self.projection_function(_bregman_map(M, lambda_k, subgrad_lambda_k))

    def dual_step(self):
        ###############
        # Preparation #
        ###############

        # if it's the first iteration, we have to make an oracle call to fill the subgradient and the d_k
        # for lambda_0; the algorithm assumes that these quantities are known for each iterate (including 0-th)
        # if not self.diff_d_k:
        if self.iteration_number == 1:
            self.x_hat_k, self.d_hat_k, self.diff_d_hat_k = self.oracle(self.lambda_hat_k)
            self.oracle_calls += 1
            self.lambda_k = self.lambda_hat_k
            self.d_k = self.d_hat_k
            self.diff_d_k = self.diff_d_hat_k
            self.notify_observers()

        ##############################
        # Step 1 (see (2.16) in [1]) #
        ##############################

        i_k = 0
        smallest_i_k_found = 0

        while not smallest_i_k_found:
            # find next test point
            lambda_k_plus = self._bregman_map(2 ** i_k * self.L_k, self.lambda_hat_k, self.diff_d_hat_k)
            # query oracle at test point
            x_k_plus, d_k_plus, diff_d_k_plus, = self.oracle(lambda_k_plus)
            self.oracle_calls += 1

            # check condition given in the inequality of Step 1.
            if (-d_k_plus <= -self.d_hat_k
                             + np.dot(-self.diff_d_hat_k, lambda_k_plus - self.lambda_hat_k)
                             + 2**(i_k-1)*self.L_k*(np.linalg.norm(lambda_k_plus-self.lambda_hat_k, 2)**2)
                             + 0.5*self.epsilon):
                smallest_i_k_found = 1
            else:
                i_k += 1

        ##########
        # Step 2 #
        ##########

        self.iteration_number += 1
        self.L_k = 2**(i_k-1)*self.L_k

        # -- Averaging -- Synthesize outputs
        self.S_k += float(1)/float(self.L_k)
        self.sum_lambda_tilde_k += float(1) / float(self.L_k) * self.lambda_hat_k
        self.lambda_tilde_k = float(1) / float(self.S_k) * self.sum_lambda_tilde_k
        self.sum_d_tilde_k += float(1) / float(self.L_k) * self.d_hat_k
        self.d_tilde_k = float(1) / float(self.S_k) * self.sum_d_tilde_k

        # self.S_k += self.L_k
        # self.sum_lambda_tilde_k += self.L_k * self.lambda_hat_k
        # self.lambda_tilde_k = float(1) / float(self.S_k) * self.sum_lambda_tilde_k
        # self.sum_d_tilde_k += self.L_k * self.d_hat_k
        # self.d_tilde_k = float(1) / float(self.S_k) * self.sum_d_tilde_k
        # -- Averaging --

        # Update
        self.lambda_hat_k = lambda_k_plus

        # and for record keeping...
        self.d_hat_k = d_k_plus
        self.diff_d_hat_k = diff_d_k_plus
        self.x_hat_k = x_k_plus
        self.i_k = i_k

        # Calculate ouputs depending on whether averaging is active or not:
        if self.averaging:
            # we have an additional oracle call
            # projection here would not be required technically, but because of numerics when constructing the convex
            # combination, we call it
            self.lambda_k = self.projection_function(self.lambda_tilde_k)
            self.x_k, self.d_k, self.diff_d_k = self.oracle(self.lambda_k)
            self.oracle_calls += 1
        else:
            self.x_k = self.x_hat_k
            self.d_k = self.d_hat_k
            self.lambda_k = self.lambda_hat_k
            self.diff_d_k = self.diff_d_hat_k

        # log signal to any observers connected
        self.notify_observers()


##################################
# UNIVERSAL DUAL GRADIENT METHOD #
##################################

class UniversalDGM(DualMethod, Observable):
    """
    Implementation of Algorithm (3.2) in [1], the Universal Dual Gradient Method.
    Note that the algorithm is written for the maximization of a convex function, while in the duality
    framework we maximize a concave fct. Hence, f(x) := -d(lambda)
    """
    def __init__(self, oracle, projection_function, dimension=0, epsilon=UGM_DEFAULT_EPSILON, averaging=False):
        super(UniversalDGM, self).__init__()

        self.desc = 'UDGM, $\epsilon = {}$'.format(epsilon)

        self.oracle = oracle
        self.projection_function = projection_function

        self.iteration_number = 1
        self.oracle_calls = 0

        # self.d_k = np.zeros(1, dtype=float)
        if dimension == 0:
            self.lambda_hat_k = self.projection_function(0)
            self.dimension = len(self.lambda_hat_k)
        else:
            self.dimension = dimension
            self.lambda_hat_k = self.projection_function(np.zeros(self.dimension, dtype=float))
        # self.dimension = dimension
        # self.lambda_k = self.projection_function(np.zeros(self.dimension, dtype=float))

        # # Init of the method
        # # if it's the first iteration, we have to make an oracle call to fill the subgradient and the d_k
        # # for lambda_0; the algorithm assumes that these quantities are known for each iterate (including 0-th)
        # # if self.iteration_number == 1:
        # self.x_k, self.d_k, self.diff_d_k = self.oracle(self.lambda_k)
        # self.oracle_calls += 1
        self.x_hat_k = 0
        self.d_hat_k = 0
        self.diff_d_hat_k = 0

        # specific to U-DGM
        self.L_k = float(UGM_DEFAULT_L_0)  # if you use something else, make sure it's a float!
        self.epsilon = float(epsilon)
        self.i_k = 0
        self.phi_k = copy.deepcopy(self.lambda_hat_k)

        # -- Averaging -- Synthesize outputs
        # Variables to synthesize solution from algorithm's process
        # records of d_tilda_k and lambda_tilda_k ("averages") according to Eqns. below 2.17
        self.S_k = float(1)/float(self.L_k)
        self.lambda_tilde_k = copy.deepcopy(self.lambda_hat_k)
        self.sum_lambda_tilde_k = copy.deepcopy(self.lambda_hat_k)  # \sum_i=0^k lambda_tilda_k
        self.d_tilde_k = 0
        self.sum_d_tilde_k = 0

        self.averaging = averaging
        if self.averaging:
            self.lambda_k = self.lambda_tilde_k
            self.d_k = self.d_tilde_k
            self.diff_d_k = self.diff_d_hat_k
            self.x_k = self.x_hat_k
        else:
            self.lambda_k = self.lambda_hat_k
            self.d_k = self.d_hat_k
            self.diff_d_k = self.diff_d_hat_k
            self.x_k = self.x_hat_k
        # -- Averaging --

        # for record keeping
        self.method_name = 'UDGM'
        self.parameter = epsilon

    def _bregman_map(self, M, lambda_k, subgrad_lambda_k):
        return self.projection_function(_bregman_map(M, lambda_k, subgrad_lambda_k))

    def dual_step(self):
        # Implementation of Algorithm (3.2) in [1], the Universal Dual Gradient Method.

        if self.iteration_number == 1:
            # Init
            # if it's the first iteration, we have to make an oracle call to fill the subgradient and the d_k
            # for lambda_0; the algorithm assumes that these quantities are known for each iterate (including 0-th)
            # if self.iteration_number == 1:
            self.x_hat_k, self.d_hat_k, self.diff_d_hat_k = self.oracle(self.lambda_hat_k)
            self.oracle_calls += 1
            self.lambda_k = self.lambda_hat_k
            self.d_k = self.d_hat_k
            self.diff_d_k = self.diff_d_hat_k
            self.notify_observers()

        ##########
        # Step 0 #
        ##########

        i_k = 0
        smallest_i_k_found = 0

        while not smallest_i_k_found:
            # first, calculate lambda_k_ik (test point)
            lambda_k_ik = self.phi_k + float(1.0)/(2**i_k*self.L_k)*self.diff_d_hat_k
            lambda_k_ik = self.projection_function(lambda_k_ik)

            # then, call oracle at lambda_k_ik (test point)
            x_k_ik, d_k_ik, diff_d_k_ik, = self.oracle(lambda_k_ik)
            self.oracle_calls += 1

            # before I can test the condition I have to calculate the Bregman point, and invoke once again the oracle
            # to evaluate d(bregman(lambda_k_ik))
            bregman_lambda_k_ik = self._bregman_map(2**i_k*self.L_k, lambda_k_ik, diff_d_k_ik)
            bregman_x_k_ik, bregman_d_k_ik, bregman_subgrad_lambda_k_ik, = self.oracle(bregman_lambda_k_ik)
            self.oracle_calls += 1

            # then test condition
            if (-bregman_d_k_ik <= -d_k_ik
                                    + np.dot(-diff_d_k_ik, bregman_lambda_k_ik - lambda_k_ik)
                                    + float(2**i_k*self.L_k)/float(2)*(np.linalg.norm(lambda_k_ik - bregman_lambda_k_ik, 2)**2)
                                    + float(self.epsilon)/float(2)):
                smallest_i_k_found = 1
            else:
                i_k += 1

        ##########
        # Step 2 #
        ##########

        self.iteration_number += 1
        self.L_k = 2**(i_k-1)*self.L_k

        # -- Averaging -- Synthesize outputs
        self.S_k += float(1)/float(self.L_k)
        self.sum_lambda_tilde_k += float(1) / float(self.L_k) * bregman_lambda_k_ik
        self.lambda_tilde_k = float(1) / float(self.S_k) * self.sum_lambda_tilde_k
        self.sum_d_tilde_k += float(1) / float(self.L_k) * bregman_d_k_ik
        self.d_tilde_k = float(1) / float(self.S_k) * self.sum_d_tilde_k
        # -- Averaging --

        self.lambda_hat_k = lambda_k_ik
        self.phi_k += float(1.0)/(2*self.L_k)*self.diff_d_hat_k
        # and for the record ...
        self.d_hat_k = d_k_ik
        self.diff_d_hat_k = diff_d_k_ik
        self.x_hat_k = x_k_ik
        self.i_k = i_k

        # Calculate ouputs depending on whether averaging is active or not:
        if self.averaging:
            # we have an additional oracle call
            # projection here would not be required technically, but because of numerics when constructing the convex
            # combination, we call it
            self.lambda_k = self.projection_function(self.lambda_tilde_k)
            self.x_k, self.d_k, self.diff_d_k = self.oracle(self.lambda_k)
            self.oracle_calls += 1
        else:
            self.x_k = self.x_hat_k
            self.d_k = self.d_hat_k
            self.lambda_k = self.lambda_hat_k
            self.diff_d_k = self.diff_d_hat_k

        # log signal to any observers connected
        self.notify_observers()


##################################
# UNIVERSAL FAST GRADIENT METHOD #
##################################

class UniversalFGM(DualMethod, Observable):
    """
    Implementation of Algorithm (4.1) in [1], the Universal Fast Gradient Method.
    Note that the algorithm is written for the maximization of a convex function, while in the duality
    framework we maximize a concave fct. Hence, f(x) := -d(lambda)
    """
    def __init__(self, oracle, projection_function, dimension=0, epsilon=UGM_DEFAULT_EPSILON, averaging=False):
        super(UniversalFGM, self).__init__()

        self.desc = 'UFGM, $\epsilon = {}$'.format(epsilon)

        self.oracle = oracle
        self.projection_function = projection_function

        self.iteration_number = 1
        self.oracle_calls = 0

        self.d_hat_k = np.zeros(1, dtype=float)
        if dimension == 0:
            self.lambda_hat_k = self.projection_function(0)
            self.dimension = len(self.lambda_hat_k)
        else:
            self.dimension = dimension
            self.lambda_hat_k = self.projection_function(np.zeros(self.dimension, dtype=float))
        self.x_hat_k = 0

        # specific to U-PGM
        self.diff_d_hat_k = 0
        self.L_k = float(UGM_DEFAULT_L_0)  # if you use something else, make sure it's a float!
        self.epsilon = float(epsilon)
        self.i_k = 0
        self.phi_k = copy.deepcopy(self.lambda_hat_k)

        self.y_k = copy.deepcopy(self.lambda_hat_k)
        self.A_k = 0
        self.a_k = copy.deepcopy(self.lambda_hat_k)
        self.tau_k = 0
        self.v_k = 0

        self.averaging = averaging
        if self.averaging:
            self.lambda_k = self.y_k
            self.d_k = self.d_hat_k
            self.diff_d_k = self.diff_d_hat_k
            self.x_k = self.x_hat_k
        else:
            self.lambda_k = self.lambda_hat_k
            self.d_k = self.d_hat_k
            self.diff_d_k = self.diff_d_hat_k
            self.x_k = self.x_hat_k

        # for record keeping
        self.method_name = 'UFGM'
        self.parameter = epsilon

    def _bregman_map(self, M, lambda_k, subgrad_lambda_k):
        return self.projection_function(_bregman_map(M, lambda_k, subgrad_lambda_k))

    def dual_step(self):
        ##########
        # Step 1 #
        ##########

        # find v_k
        v_k = self.projection_function(self.phi_k)

        ##########
        # Step 2 #
        ##########

        smallest_i_k_found = 0
        i_k = 0

        while not smallest_i_k_found:
            a_kp_ik = float(1 + np.sqrt(1+self.A_k*2**(i_k+2)*self.L_k))/float(2**(i_k+1)*self.L_k)
            A_kp_ik = self.A_k + a_kp_ik
            tau_k_ik = float(a_kp_ik)/float(A_kp_ik)
            # Find test point
            lambda_kp_ik = tau_k_ik*v_k + (1-tau_k_ik)*self.y_k
            # Query oracle at test point
            x_kp_ik, d_kp_ik, diff_kp_ik, = self.oracle(lambda_kp_ik)
            self.oracle_calls += 1
            # Continue with the computations
            hat_lambda_kp_ik = v_k + a_kp_ik*diff_kp_ik
            hat_lambda_kp_ik = self.projection_function(hat_lambda_kp_ik)
            y_kp_ik = tau_k_ik*hat_lambda_kp_ik + (1-tau_k_ik)*self.y_k
            # Query oracle again at y_kp_ik
            x_y_kp_ik, d_y_kp_ik, diff_y_kp_ik, = self.oracle(y_kp_ik)
            self.oracle_calls += 1
            # Test condition
            if -d_y_kp_ik <= (-d_kp_ik
                            + np.dot(-diff_kp_ik,y_kp_ik-lambda_kp_ik)
                            + 2**(i_k-1)*self.L_k*(np.linalg.norm(y_kp_ik - lambda_kp_ik,2)**2)
                            + float(self.epsilon)/float(2.0)*tau_k_ik):
                smallest_i_k_found = 1
            else:
                i_k += 1

        ##########
        # Step 3 #
        ##########

        self.iteration_number += 1
        # Perform step
        self.lambda_hat_k = lambda_kp_ik
        self.y_k = y_kp_ik
        self.a_k = a_kp_ik
        self.tau_k = tau_k_ik
        self.A_k = self.A_k + self.a_k
        self.L_k = 2**(i_k-1)*self.L_k
        self.phi_k += self.a_k * self.diff_d_hat_k

        # Record additional information about iterate
        self.d_hat_k= d_kp_ik
        self.diff_d_hat_k = diff_kp_ik
        self.x_hat_k = x_kp_ik
        self.i_k = i_k

        # Calculate ouputs depending on whether averaging is active or not:
        if self.averaging:
            # we have an additional oracle call
            self.lambda_k = self.y_k
            self.x_k, self.d_k, self.diff_d_k = self.oracle(self.lambda_k)
            self.oracle_calls += 1
        else:
            self.x_k = self.x_hat_k
            self.d_k = self.d_hat_k
            self.lambda_k = self.lambda_hat_k
            self.diff_d_k = self.diff_d_hat_k

        # log signal to any observers connected
        self.notify_observers()
