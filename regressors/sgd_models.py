from utils import get_data

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error

import sys


def sgd_regression(data_tuple, out_filename, n_splits=5, verbosity=2):
    sys.stdout = open(out_filename + '.out', 'w')

    x_train, x_test, y_train, y_test = get_data(data_tuple)

    clf = SGDRegressor(verbose=verbosity)
    clf.fit(x_train, y_train)

    train_r2 = clf.score(x_train, y_train)
    all_r2 = r2_score(clf.predict(np.vstack((x_train, x_test))), np.concatenate((y_train, y_test)))
    train_mse = mean_squared_error(clf.predict(x_train), y_train)
    test_mse = mean_squared_error(clf.predict(x_test), y_test)
    all_mse = mean_squared_error(clf.predict(np.vstack((x_train, x_test))), np.concatenate((y_train, y_test)))
    print('train_r2: ', train_r2)
    print('all_r2: ', all_r2)
    print('diff_r2: ', train_r2 - all_r2)
    print('train_mse: ', train_mse)
    print('test_mse: ', test_mse)
    print('all_mse: ', all_mse)
    print('predicted: ', clf.predict(x_test[-1].reshape(1, -1))[0])
    print('actual: ', y_test[-1])