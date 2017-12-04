from __future__ import print_function
from __future__ import division
from nsopy.method_loggers import Observable
from nsopy.base import DualMethod


#############################################
# TEMPLATE OF CONCRETE CLASS IMPLEMENTATION #
#############################################

class TemplateMethod(DualMethod, Observable):
    """ Implementation of a dual method """
    def __init__(self, oracle, projection_function):
        super(TemplateMethod, self).__init__()
        self.oracle = oracle
        self.projection_function = projection_function
        self.x = 0
        self.oracle_calls = 0
        self.desc = 'template method'

    def dual_step(self):
        print('oracle: ' + str(self.oracle))
        print('proj: ' + str(self.projection_function))
        print('and we notify observers of the step')
        self.x += 1
        self.notify_observers()