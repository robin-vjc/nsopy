class SolutionMethod(object):
    """ Interface for all the nsopy implemented """
    def dual_step(self):
        raise NotImplementedError()

    def step(self):
        self.dual_step()