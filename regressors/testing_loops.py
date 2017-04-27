from tpot_models import tpot_regression
from sgd_models import sgd_regression
from mlp_models import mlp_regression

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


def tpot_loop(sample=False, no_tests=3, generations=10, pop_size=100):
    if sample:
        trials = tpot_variations(no_tests=no_tests)
    else:
        trials = tpot_variations(no_tests=no_tests)
    for params in trials:
        data_tuple = params[:-1] + (False, sample)
        #if sample:
        #    dirpath = 'tpot_results/test'
        #    filepath = 'tpot_results/test/{}'.format(params[-1])
        #else:
        dirpath = 'tpot_results/{}/{}/{}'.format(*params)
        filepath = 'tpot_results/{}/{}/{}/{}'.format(*params)
        os.makedirs(dirpath, exist_ok=True)

        tpot_regression(data_tuple, filepath, generations=generations, pop_size=pop_size)


def sgd_loop(sample=False, no_tests=3):
    if sample:
        trials = tpot_variations(no_tests=no_tests)
    else:
        trials = tpot_variations(no_tests=no_tests)
    for params in trials:
        data_tuple = params[:-1] + (False, sample)
        #if sample:
        #    dirpath = 'tpot_results/test'
        #    filepath = 'tpot_results/test/{}'.format(params[-1])
        #else:
        dirpath = 'sgd_results/{}/{}/{}'.format(*params)
        filepath = 'sgd_results/{}/{}/{}/{}'.format(*params)
        os.makedirs(dirpath, exist_ok=True)

        sgd_regression(data_tuple, filepath)


def mlp_loop(sample=False, no_tests=3):
    if sample:
        trials = tpot_variations(no_tests=no_tests)
    else:
        trials = tpot_variations(no_tests=no_tests)
    for params in trials:
        data_tuple = params[:-1] + (False, sample)
        #if sample:
        #    dirpath = 'tpot_results/test'
        #    filepath = 'tpot_results/test/{}'.format(params[-1])
        #else:
        dirpath = 'mlp_results/{}/{}/{}'.format(*params)
        filepath = 'mlp_results/{}/{}/{}/{}'.format(*params)
        os.makedirs(dirpath, exist_ok=True)

        sgd_regression(data_tuple, filepath)

if __name__ == '__main__':
    if '-tpot' in sys.argv:
        if '-g' in sys.argv:
            g = int(sys.argv[sys.argv.index('-g') + 1])
        else:
            g = 10
        if '-p' in sys.argv:
            p = int(sys.argv[sys.argv.index('-p') + 1])
        else:
            p = 100
        if '-n' in sys.argv:
            n = int(sys.argv[sys.argv.index('-n') + 1])
        else:
            n = 3
        if '-a' in sys.argv:
            tpot_loop(no_tests=n, generations=g, pop_size=p)
        elif '-l' in sys.argv:
            tpot_loop(True, no_tests=n, generations=g, pop_size=p)
    elif '-sgd' in sys.argv:
        if '-n' in sys.argv:
            n = int(sys.argv[sys.argv.index('-n') + 1])
        else:
            n = 3
        if '-a' in sys.argv:
            sgd_loop(no_tests=n)
        elif '-l' in sys.argv:
            sgd_loop(True, no_tests=n)
    elif '-mlp' in sys.argv:
        if '-n' in sys.argv:
            n = int(sys.argv[sys.argv.index('-n') + 1])
        else:
            n = 3
        if '-a' in sys.argv:
            mlp_loop(no_tests=n)
        elif '-l' in sys.argv:
            mlp_loop(True, no_tests=n)

    else:
        print('USAGE STATEMENT: \n'
              '-tpot\n'
              '     -a for all | -l for limited | -n # for n trials\n'
              '     -g # -p # for generations and popsize, respectively\n'
              '-sgd\n'
              '     -a for all | -l for limited | -n # for n trials\n'
              '-mlp\n'
              '     -a for all | -l for limited | -n # for n trials\n')
