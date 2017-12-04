# Implementation of the observer pattern, to allow (optional) recording of algorithm progress
import copy
import time


class Observable(object):
    """ Make object observable. """
    def __init__(self):
        self.observers = []

    def register_observer(self, observer):
        self.observers.append(observer)

    def remove_observer(self, observer):
        self.observers.remove(observer)

    def notify_observers(self):
        for observer in self.observers:
            observer.update()


class Observer(object):
    def update(self):
        """
        This method is called whenever an Observable object calls notify_observers()
        (usually when the Observable object changes state).
        """
        raise NotImplementedError()


############################
# CONCRETE IMPLEMENTATIONS #
############################

class TemplateMethodLogger(Observer):
    def __init__(self, template_method):
        # we get a reference to the method we are supposed to
        self.method = template_method
        # we need to register the logger with the observable, so we get the notifications
        self.method.register_observer(self)
        self.x_iterates = []

    def update(self):
        # what we do when method sends updates
        self.x_iterates.append(self.method.x)


class GenericDualMethodLogger(Observer):
    """
    Works with all implemented dual nsopy, and logs only variables that are common across all
    of them (lambda_k, d_k, etc)
    """
    def __init__(self, dual_method):
        # we get a reference to the method we are supposed to
        self.method = dual_method
        # we need to register the logger with the observable, so we get the notifications
        self.method.register_observer(self)
        self.lambda_k_iterates = []
        self.d_k_iterates = []
        self.x_k_iterates = []

    def update(self):
        # what we do when method sends updates
        self.lambda_k_iterates.append(copy.copy(self.method.lambda_k))
        self.d_k_iterates.append(copy.copy(self.method.d_k))
        self.x_k_iterates.append(copy.copy(self.method.x_k))


class EnhancedDualMethodLogger(Observer):
    """
    Additionally logs # of oracle calls, time
    """
    def __init__(self, dual_method):
        # we get a reference to the method we are supposed to
        self.method = dual_method
        # we need to register the logger with the observable, so we get the notifications
        self.method.register_observer(self)
        self.lambda_k_iterates = []
        self.d_k_iterates = []
        self.x_k_iterates = []
        self.start_time = 0
        self.iteration_time = []
        self.oracle_calls = []

        # -- TEMP
        # self.L_k_iterates = []
        # -- TEMP

    def update(self):
        # what we do when method sends updates
        if not self.start_time:
            self.start_time = time.time()
        self.iteration_time.append(time.time() - self.start_time)

        self.oracle_calls.append(copy.copy(self.method.oracle_calls))

        self.lambda_k_iterates.append(copy.copy(self.method.lambda_k))
        self.d_k_iterates.append(copy.copy(self.method.d_k))
        self.x_k_iterates.append(copy.copy(self.method.x_k))

        # -- TEMP
        # self.L_k_iterates.append(copy.copy(self.method.L_k))
        # -- TEMP


class DualDgmFgmMethodLogger(Observer):
    """
    Additionally logs # of oracle calls, time
    """
    def __init__(self, dual_method):
        # we get a reference to the method we are supposed to
        self.method = dual_method
        # we need to register the logger with the observable, so we get the notifications
        self.method.register_observer(self)
        self.lambda_k_iterates = []
        self.d_k_iterates = []
        self.x_k_iterates = []
        self.start_time = 0
        self.iteration_time = []
        self.oracle_calls = []
        self.L_k_iterates = []

    def update(self):
        # what we do when method sends updates
        if not self.start_time:
            self.start_time = time.time()
        self.iteration_time.append(time.time() - self.start_time)

        self.oracle_calls.append(copy.copy(self.method.oracle_calls))

        self.lambda_k_iterates.append(copy.copy(self.method.lambda_k))
        self.d_k_iterates.append(copy.copy(self.method.d_k))
        self.x_k_iterates.append(copy.copy(self.method.x_k))
        self.L_k_iterates.append(copy.copy(self.method.L_k))


class PGMVisualizationLogger(Observer):
    """
    Additionally logs # of oracle calls, time
    """
    def __init__(self, dual_method):
        # we get a reference to the method we are supposed to
        self.method = dual_method
        # we need to register the logger with the observable, so we get the notifications
        self.method.register_observer(self)
        self.lambda_k_iterates = []
        self.d_k_iterates = []
        self.x_k_iterates = []
        self.start_time = 0
        self.iteration_time = []
        self.oracle_calls = []
        self.L_k_iterates = []
        self.lambda_tilde_k = []
        self.d_tilde_k = []

    def update(self):
        # what we do when method sends updates
        if not self.start_time:
            self.start_time = time.time()
        self.iteration_time.append(time.time() - self.start_time)

        self.oracle_calls.append(copy.copy(self.method.oracle_calls))

        self.lambda_k_iterates.append(copy.copy(self.method.lambda_k))
        self.d_k_iterates.append(copy.copy(self.method.d_k))
        self.x_k_iterates.append(copy.copy(self.method.x_k))
        self.L_k_iterates.append(copy.copy(self.method.L_k))
        self.lambda_tilde_k.append(copy.deepcopy(self.method.lambda_tilde_k))
        self.d_tilde_k.append(copy.deepcopy(self.method.d_tilde_k))


class SlimDualMethodLogger(Observer):
    """
    Don't store lambda_k nor x_k. For big problems.
    """
    def __init__(self, dual_method):
        # we get a reference to the method we are supposed to
        self.method = dual_method
        # we need to register the logger with the observable, so we get the notifications
        self.method.register_observer(self)
        self.d_k_iterates = []
        self.start_time = []
        self.iteration_time = []
        self.oracle_calls = []

    def update(self):
        # what we do when method sends updates
        self.start_time.append(time.time())
        # if it's not the first round, we record difference from this iteration to previous
        if self.d_k_iterates:
            self.iteration_time.append(self.start_time[-1] - self.start_time[-2])
        self.oracle_calls.append(copy.copy(self.method.oracle_calls))
        self.d_k_iterates.append(copy.copy(self.method.d_k))