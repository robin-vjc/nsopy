from examples.analytical_oracles import SecondAnalyticalExampleInnerProblem
from methods.methods_factory import DualMethodsFactory, AVAILABLE_METHODS
from methods.method_loggers import SlimDualMethodLogger
from methods.utils import record_logger


def test_slimlogger_recorder():
    # instantiate ip (first instance of 2 stage)
    # ip = BenchmarkInnerProblemsFactory(type='supply chains', subtype='all', instance_n=0)
    # ip = BenchmarkInnerProblemsFactory(type='2 stage stoch', subtype='dcap', instance_n=0)
    ip = SecondAnalyticalExampleInnerProblem()

    # method, logger
    method = DualMethodsFactory(ip, 'SG 1/k', 1.5)
    logger = SlimDualMethodLogger(method)

    N_ORACLE_CALLS = 3

    # run
    while method.oracle_calls < N_ORACLE_CALLS:
        method.dual_step()

    # execute recorder
    record_logger(logger)