from utils import get_data

import numpy as np
from tpot import TPOTRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.externals import joblib

import sys
import pickle


def tpot_regression(data_tuple, out_filename, n_splits=3,
                    generations=5, pop_size=50, verbosity=2):
    # sys.stdout = open(out_filename + '.out', 'w')

    x_train, x_test, y_train, y_test = get_data(data_tuple)
    """
    print('xtr', x_train.shape, x_train.dtype, np.isnan(np.min(x_train)))
    print('xte', x_test.shape, x_test.dtype, np.isnan(np.min(x_test)))
    print('ytr', y_train.shape, y_train.dtype, np.isnan(np.min(y_train)))
    print('yte', y_test.shape, y_test.dtype, np.isnan(np.min(y_test)))
    """
    tpot = TPOTRegressor(generations=generations, population_size=pop_size,
                         verbosity=verbosity, cv=TimeSeriesSplit(n_splits))
    tpot.fit(x_train, y_train)
    print(tpot.score(x_train, y_train))
    print(type(tpot))
    # with open(out_filename + '.pkl', 'w') as f:
    #     pickle.dump(tpot, f)
    # tpot.export(out_filename + '_pipeline.py')


if __name__ == '__main__':
    data = tuple(get_data(None))
    tpot_regression(None, 'test', generations=1, pop_size=10)