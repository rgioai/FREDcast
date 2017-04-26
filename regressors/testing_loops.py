from tpot_models import tpot_regression

import itertools
import os
import sys


def tpot_variations(y=True, norm=True, residual=True, no_tests=3):
    if y:
        combins = [['gdp', 'cpi', 'payroll', 'unemployment']]
    else:
        combins = [['gdp']]
    if norm:
        combins.append(['normal_dist', 'percent_change', 'zero_one'])
    else:
        combins.append(['zero_one'])
    if residual:
        combins.append(['exp_residual', 'gdp_residual', 'linear_residual', 'none'])
    else:
        combins.append(['none'])
    if no_tests > 1:
        tests = []
        for i in range(1, no_tests+1):
            tests.append('test%d' % i)
        combins.append(tests)
    else:
        combins.append(['test0'])
    return itertools.product(*combins)


def tpot_loop(sample=False, no_tests=3):
    if sample:
        trials = tpot_variations(False, False, False, no_tests=no_tests)
    else:
        trials = tpot_variations(no_tests=no_tests)
    for params in trials:
        data_tuple = params[:-1] + (False, sample)
        if sample:
            dirpath = 'tpot_results/test'
            filepath = 'tpot_results/test/{}'.format(params[-1])
        else:
            dirpath = 'tpot_results/{}/{}/{}'.format(*params)
            filepath = 'tpot_results/{}/{}/{}/{}'.format(*params)
        os.makedirs(dirpath, exist_ok=True)

        tpot_regression(data_tuple, filepath)

if __name__ == '__main__':
    if '-tpot' in sys.argv:
        if '-a' in sys.argv:
            tpot_loop()
        elif '-l' in sys.argv:
            tpot_loop(True)
        elif '-n' in sys.argv:
            n = int(sys.argv[sys.argv.index('-n') + 1])
            tpot_loop(no_tests=n)
    else:
        print('USAGE STATEMENT: -tpot\n     -a for all | -l for limited | -n # for n trials on all')
