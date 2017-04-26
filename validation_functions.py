import h5py
import numpy as np


def check_for_nan(hdf5_filepath):
    hdf5 = h5py.File(hdf5_filepath, 'r')
    paths = []

    def find_datasets(name):
        if '/' in name:
            if 'data/raw' not in name and 'admin/dates_index' not in name and name != 'data/norm_data':
                paths.append(name)
    hdf5.visit(find_datasets)

    for path in paths:
        dset = np.asarray(hdf5[path])
        if np.any(np.isnan(dset)):
            print(str(path) + ' in ' + str(hdf5_filepath) + ' contains NaN.')
    hdf5.close()


def check_dsets(hdf5_filepath):
    hdf5 = h5py.File(hdf5_filepath)
    train_x_paths = []
    train_y_paths = []
    test_x_paths = []
    test_y_paths = []

    def find_datasets(name):
        if '/train_x' in name:
            train_x_paths.append(name)
        elif '/train_x' in name:
            train_x_paths.append(name)
        elif '/train_y' in name:
            train_y_paths.append(name)
        elif '/test_x' in name:
            test_x_paths.append(name)
        elif '/test_y' in name:
            test_y_paths.append(name)

    hdf5.visit(find_datasets)

    date_dset = np.asarray(hdf5['admin/dates_index'])

    for path in train_x_paths:
        dset = np.asarray(hdf5[path])
        if dset.shape[0] != 3:
            print('Shape for ' + str(path) + ' in ' + str(hdf5_filepath) + ' is invalid.')
    for path in train_y_paths:
        dset = np.asarray(hdf5[path])
        if dset.shape[0] != 3:
            print('Shape for ' + str(path) + ' in ' + str(hdf5_filepath) + ' is invalid.')
    for path in test_x_paths:
        dset = np.asarray(hdf5[path])
        if dset.shape[0] != date_dset.shape[0] - 3:
            print('Shape for ' + str(path) + ' in ' + str(hdf5_filepath) + ' is invalid.')
    for path in test_y_paths:
        dset = np.asarray(hdf5[path])
        if dset.shape[0] != date_dset.shape[0] - 3:
            print('Shape for ' + str(path) + ' in ' + str(hdf5_filepath) + ' is invalid.')
    hdf5.close()


if __name__ == '__main__':
    check_for_nan('FREDcast_sample.hdf5')
    check_for_nan('FREDcast.hdf5')
    check_for_nan('split_data_sample.hdf5')
    check_for_nan('split_data.hdf5')
    check_for_nan('rnn_data_sample.hdf5')
    check_for_nan('rnn_data.hdf5')

    check_dsets('split_data_sample.hdf5')
    check_dsets('split_data.hdf5')
    check_dsets('rnn_data_sample.hdf5')
    check_dsets('rnn_data.hdf5')
