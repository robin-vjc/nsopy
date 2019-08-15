import datetime
import pandas as pd
from ast import literal_eval


def invert_oracle_sense(oracle):
    def inverted_oracle(lambda_k):
        x_k, d_k, diff_d_k = oracle(lambda_k)

        return x_k, -d_k, -diff_d_k

    return inverted_oracle


def record_logger(logger, filename=r'logger_record.csv'):
    """ Records the information contained in a method logger into a csv. """
    inner_problem = logger.method.oracle.__self__

    instance_name = inner_problem.instance_name
    instance_subtype = inner_problem.instance_subtype
    instance_type = inner_problem.instance_type
    method_desc = logger.method.desc
    method_name = logger.method.method_name
    method_parameter = logger.method.parameter
    d_k = logger.d_k_iterates
    d_k = ['%.2f' % elem for elem in d_k]  # format it
    oracle_calls = logger.oracle_calls
    computation_times = logger.iteration_time
    computation_times = ['%.2f' % elem for elem in computation_times] # format it
    date = datetime.datetime.now()

    try:
        df = pd.read_csv(filename)
    except IOError as e:
        # TODO there is something buggy about this when I run it in the ipython notebook; it thinks the file int created
        print(e)
        print('Record file does not exist. Creating {} ...'.format(filename))

        columns = ('date', 'instance_name', 'instance_subtype', 'instance_type',
                   'method_desc', 'method_name', 'method_parameter', 'd_k', 'oracle_calls',
                   'computation_times')
        df = pd.DataFrame(columns=columns)

    # Append data to dataframe
    df.loc[len(df)] = [
        date, instance_name, instance_subtype, instance_type,
        method_desc, method_name, method_parameter,
        d_k, oracle_calls, computation_times
    ]

    df.to_csv(filename, index=False)


def flatten_record_dataframe(df):
    """ The above function stores
     - d_k
     - oracle_calls
     - computation_times
     as strings. We flatten these back to lists (of floats) """
    for index, row in df.iterrows():
        d_k = literal_eval(row.d_k)  # list results were saved as strings...
        d_k = [float(i) for i in d_k]  # so we have to convert them back to floats manually
        df.set_value(index, 'd_k', d_k)

        # same for oracle_calls
        oracle_calls = literal_eval(row.oracle_calls)
        oracle_calls = [float(i) for i in oracle_calls]
        df.set_value(index, 'oracle_calls', oracle_calls)

        # and computation_times
        computation_times = literal_eval(row.computation_times)  # list results were saved as strings...
        computation_times = [float(i) for i in computation_times]  # so we have to convert them back to floats manually
        df.set_value(index, 'computation_times', computation_times)

    return df