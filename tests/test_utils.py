import time

from nsopy.loggers import SlimDualMethodLogger
from nsopy.methods_factory import DualMethodsFactory
from tests.analytical_oracles import SecondAnalyticalExampleInnerProblem


def test_slimlogger_recorder():
    ip = SecondAnalyticalExampleInnerProblem()

    # method, logger
    method = DualMethodsFactory(ip, 'SG 1/k', 1.5)
    logger = SlimDualMethodLogger(method)

    N_ORACLE_CALLS = 3

    # run
    while method.oracle_calls < N_ORACLE_CALLS:
        method.dual_step()

    assert len(logger.d_k_iterates) == 3
    assert type(time.localtime(logger.start_time[0]).tm_year) == int
