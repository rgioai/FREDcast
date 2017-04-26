from utils import get_data

import numpy as np
from tpot import TPOTRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.externals import joblib
from sklearn.metrics import r2_score, mean_squared_error

import sys
import pickle


def tpot_regression(data_tuple, out_filename, n_splits=5,
                    generations=10, pop_size=100, verbosity=2):
    sys.stdout = open(out_filename + '.out', 'w')

    x_train, x_test, y_train, y_test = get_data(data_tuple)
    tpot = TPOTRegressor(generations=generations, population_size=pop_size,
                         verbosity=verbosity, cv=TimeSeriesSplit(n_splits),
                         scoring='r2')
    try:
        tpot.fit(x_train, y_train)
    except ValueError as e:
        print(e)
        print(x_train.shape, y_train.shape)
        raise ValueError

    train_r2 = tpot.score(x_train, y_train)
    all_r2 = r2_score(tpot.predict(np.vstack((x_train, x_test))), np.concatenate((y_train, y_test)))
    train_mse = mean_squared_error(tpot.predict(x_train), y_train)
    test_mse = mean_squared_error(tpot.predict(x_test), y_test)
    all_mse = mean_squared_error(tpot.predict(np.vstack((x_train, x_test))), np.concatenate((y_train, y_test)))
    print('train_r2: ', train_r2)
    print('all_r2: ', all_r2)
    print('diff_r2: ', train_r2 - all_r2)
    print('train_mse: ', train_mse)
    print('test_mse: ', test_mse)
    print('all_mse: ', all_mse)
    print('predicted: ', tpot.predict(x_test[-1].reshape(1, -1))[0])
    print('actual: ', y_test[-1])
    # print('Final training score: ', tpot.score(x_train, y_train))
    # print('Final testing score: ', tpot.score(x_test, y_test))
    # with open(out_filename + '.pkl', 'w') as f:
    #     pickle.dump(tpot, f)
    tpot.export(out_filename + '_pipeline.py')


if __name__ == '__main__':
    tpot_regression(None, 'tpot_results/test/trial1', generations=1, pop_size=2)