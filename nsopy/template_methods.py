from __future__ import print_function
from __future__ import division
from nsopy.observer_pattern import Observable
from nsopy.methods.base import SolutionMethod


class TemplateMethod(SolutionMethod, Observable):
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
