import h5py
import numpy as np
import os


def aggregate_rnn_data(sample):
    cwd = os.getcwd()
    os.chdir('..')
    if sample is True:
        hdf5 = h5py.File('rnn_data_sample.hdf5')
        hdf5_fred = h5py.File('FREDcast_sample.hdf5')
    else:
        hdf5 = h5py.File('rnn_data.hdf5')
        hdf5_fred = h5py.File('FREDcast.hdf5')
    os.chdir(cwd)

    repeat_arr = np.asarray(hdf5_fred['data/norm_data/zero_one'])
    hdf5.create_dataset('admin/dates_index', data=np.asarray(hdf5_fred['admin/dates_index']))

    dset_agg = np.empty(shape=(601, repeat_arr.shape[1] * 12),
                        dtype=np.float32)
    # assert (dset_agg.shape == (601, repeat_arr.shape[1] * 12))

    filepaths = ['zero_one',
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

    loc = 0
    for path in filepaths:
        norm_dset = np.asarray(hdf5_fred['data/norm_data/' + path])
        # assert (norm_dset.shape[0] == 601)
        assert (norm_dset.dtype == np.float32)
        dset_agg[:, loc:norm_dset.shape[1]+loc] = norm_dset[:, :]
        loc += norm_dset.shape[1]

    hdf5.create_dataset('data/train_x', data=dset_agg[0:598, :])
    hdf5.create_dataset('data/test_x', data=dset_agg[-3:, :])

    y_data = [np.asarray(hdf5_fred['admin/gdp']),
              np.asarray(hdf5_fred['admin/cpi']),
              np.asarray(hdf5_fred['admin/payroll']),
              np.asarray(hdf5_fred['admin/unemployment'])]

    y_data = np.hstack(y_data)
    hdf5.create_dataset('data/train_y', data=y_data[0:598, :])
    hdf5.create_dataset('data/test_y', data=y_data[-3:, :])
    hdf5.close()
    hdf5_fred.close()

if __name__ == '__main__':
    aggregate_rnn_data(True)
    aggregate_rnn_data(False)
