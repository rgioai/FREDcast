import os

import h5py
import numpy as np


def split_data():
    hdf5 = h5py.File('split_data.hdf5')
    hdf5_fred = h5py.File('FREDcast.hdf5')

    hdf5.create_dataset('admin/sample_values_index', data=np.asarray(hdf5_fred['admin/sample_values_index']))
    hdf5.create_dataset('admin/values_index', data=np.asarray(hdf5_fred['admin/values_index']))
    hdf5.create_dataset('admin/dates_index', data=np.asarray(hdf5_fred['admin/dates_index']))

    filepaths = ['sample_zero_one',
                 'sample_zero_one_linear_residual',
                 'sample_zero_one_exp_residual',
                 'sample_zero_one_gdp_residual',
                 'sample_percent_change',
                 'sample_percent_change_linear_residual',
                 'sample_percent_change_exp_residual',
                 'sample_percent_change_gdp_residual',
                 'sample_normal_dist',
                 'sample_normal_dist_linear_residual',
                 'sample_normal_dist_exp_residual',
                 'sample_normal_dist_gdp_residual',
                 'zero_one',
                 'zero_one_linear_residual',
                 'zero_one_exp_residual',
                 'zero_one_gdp_residual',
                 'percent_change',
                 'percent_change_linear_residual',
                 'percent_change_exp_residual',
                 'percent_change_gdp_residual',
                 'normal_dist',
                 'normal_dist_linear_residual',
                 'normal_dist_exp_residual',
                 'normal_dist_gdp_residual']

    for path in filepaths:
        norm_dset = np.asarray(hdf5_fred[path])
        assert (norm_dset.shape[0] == 601)
        assert (norm_dset.dtype == np.float32)
        hdf5.create_dataset(path + '/train_x', data=norm_dset[0:597, :])
        hdf5.create_dataset(path + '/test_x', data=norm_dset[:-3, :])
    hdf5.close()
    hdf5_fred.close()


class Interface(object):
    def __init__(self, norm_fn, residual_fn, sample=False):
        if not os.path.isfile('split_data.hdf5'):
            prompt = '>'
            print('split_data.hdf5 does not exist. Type "Y" to create it, type anything else to quit.')
            answer = input(prompt)
            if answer is 'Y' or answer is 'y':
                split_data()
            else:
                quit()

        self.hdf5 = h5py.File('split_data.hdf5')
        self.norm_fn = norm_fn
        self.residual_fn = residual_fn
        self.sample = sample

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
    def train_x(self):

        if self.sample is True:
            return np.asarray(self.hdf5['sample_' + str(self.norm_fn) + '_' + str(self.residual_fn) + '/train_x'])
        else:
            return np.asarray(self.hdf5[str(self.norm_fn) + '_' + str(self.residual_fn) + '/train_x'])

    def train_y(self):

        if self.sample is True:
            return np.asarray(self.hdf5['sample_' + str(self.norm_fn) + '_' + str(self.residual_fn) + '/train_y'])
        else:
            return np.asarray(self.hdf5[str(self.norm_fn) + '_' + str(self.residual_fn) + '/train_y'])

    def test_x(self):

        if self.sample is True:
            return np.asarray(self.hdf5['sample_' + str(self.norm_fn) + '_' + str(self.residual_fn) + '/test_x'])
        else:
            return np.asarray(self.hdf5[str(self.norm_fn) + '_' + str(self.residual_fn) + '/test_x'])

    def test_y(self):

        if self.sample is True:
            return np.asarray(self.hdf5['sample_' + str(self.norm_fn) + '_' + str(self.residual_fn) + '/test_y'])
        else:
            return np.asarray(self.hdf5[str(self.norm_fn) + '_' + str(self.residual_fn) + '/test_y'])


class TFLearn_Interface(Interface):
    def train_x(self):

        if self.sample is True:
            return self.hdf5['sample_' + str(self.norm_fn) + '_' + str(self.residual_fn) + '/train_x']
        else:
            return self.hdf5[str(self.norm_fn) + '_' + str(self.residual_fn) + '/train_x']

    def train_y(self):

        if self.sample is True:
            return self.hdf5['sample_' + str(self.norm_fn) + '_' + str(self.residual_fn) + '/train_y']
        else:
            return self.hdf5[str(self.norm_fn) + '_' + str(self.residual_fn) + '/train_y']

    def test_x(self):

        if self.sample is True:
            return self.hdf5['sample_' + str(self.norm_fn) + '_' + str(self.residual_fn) + '/test_x']
        else:
            return self.hdf5[str(self.norm_fn) + '_' + str(self.residual_fn) + '/test_x']

    def test_y(self):

        if self.sample is True:
            return self.hdf5['sample_' + str(self.norm_fn) + '_' + str(self.residual_fn) + '/test_y']
        else:
            return self.hdf5[str(self.norm_fn) + '_' + str(self.residual_fn) + '/test_y']

# Test code
# sk = SKLearn_Interface('zero_one', 'linear_residual')
# sk.train_x()
