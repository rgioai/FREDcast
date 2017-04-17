import os

import h5py
import numpy as np


def split_data():
    hdf5 = h5py.File('split_data.hdf5')
    hdf5_fred = h5py.File('FREDcast.hdf5')

    hdf5.create_dataset('admin/sample_values_index', data=np.asarray(hdf5_fred['admin/sample_values_index']))
    hdf5.create_dataset('admin/values_index', data=np.asarray(hdf5_fred['admin/values_index']))
    hdf5.create_dataset('admin/dates_index', data=np.asarray(hdf5_fred['admin/dates_index']))

    filepaths = ['data/sample_zero_one',
                 'data/sample_zero_one_linear_residual',
                 'data/sample_zero_one_exp_residual',
                 'data/sample_zero_one_gdp_residual',
                 'data/sample_percent_change',
                 'data/sample_percent_change_linear_residual',
                 'data/sample_percent_change_exp_residual',
                 'data/sample_percent_change_gdp_residual',
                 'data/sample_normal_dist',
                 'data/sample_normal_dist_linear_residual',
                 'data/sample_normal_dist_exp_residual',
                 'data/sample_normal_dist_gdp_residual']

    for path in filepaths:
        norm_dset = np.asarray(hdf5_fred[path])
        assert (norm_dset.shape[0] == 601)
        assert (norm_dset.dtype == np.float32)
        hdf5.create_dataset(path + '/train_x', data=norm_dset[0:597, :])
        hdf5.create_dataset(path + '/test_x', data=norm_dset[:-3, :])


class Interface(object):
    def __init__(self, norm_fn, residual_fn, sample=False):
        if not os.path.isfile('split_data.hdf5'):
            prompt = '>'
            print('split_data.hdf5 does not exist. Type "Y" to create it, type anything else to quit.')
            answer = input(prompt)
            if answer is 'Y' or answer is 'y':
                hdf5 = h5py.File('split_data.hdf5')
                hdf5.require_group('admin')
            else:
                quit()

        self.norm_fn = norm_fn
        self.residual_fn = residual_fn
        self.sample = sample
        raise NotImplementedError

    def train_x(self):
        raise NotImplementedError

    def train_y(self):
        raise NotImplementedError

    def test_x(self):
        raise NotImplementedError

    def test_y(self):
        raise NotImplementedError

    def get_data(self):
        return self.train_x(), self.train_y(), self.test_x(), self.test_y()


class SKLearn_Interface(Interface):
    # TODO Verify that the init method of the parent class will run without an explicit call
    def train_x(self):
        hdf5 = h5py.File('split_data.hdf5')
        return np.asarray(hdf5['sample_' + str(self.norm_fn) + '_' + str(self.residual_fn) + '/train_x'])

    def train_y(self):
        hdf5 = h5py.File('split_data.hdf5')
        return np.asarray(hdf5['sample_' + str(self.norm_fn) + '_' + str(self.residual_fn) + '/train_y'])

    def test_x(self):
        hdf5 = h5py.File('split_data.hdf5')
        return np.asarray(hdf5['sample_' + str(self.norm_fn) + '_' + str(self.residual_fn) + '/test_x'])

    def test_y(self):
        hdf5 = h5py.File('split_data.hdf5')
        return np.asarray(hdf5['sample_' + str(self.norm_fn) + '_' + str(self.residual_fn) + '/test_y'])


class TFLearn_Interface(Interface):
    def train_x(self):
        hdf5 = h5py.File('split_data.hdf5')
        return hdf5['sample_' + str(self.norm_fn) + '_' + str(self.residual_fn) + '/train_x']

    def train_y(self):
        hdf5 = h5py.File('split_data.hdf5')
        return hdf5['sample_' + str(self.norm_fn) + '_' + str(self.residual_fn) + '/train_y']

    def test_x(self):
        hdf5 = h5py.File('split_data.hdf5')
        return hdf5['sample_' + str(self.norm_fn) + '_' + str(self.residual_fn) + '/test_x']

    def test_y(self):
        hdf5 = h5py.File('split_data.hdf5')
        return hdf5['sample_' + str(self.norm_fn) + '_' + str(self.residual_fn) + '/test_y']
