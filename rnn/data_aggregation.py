import h5py
import numpy as np


def aggregate_rnn_data():
    hdf5 = h5py.File('rnn_data.hdf5')
    hdf5_fred = h5py.File('FREDcast.hdf5')

    repeat_arr = np.array([np.asarray(hdf5_fred['zero_one'])] * 12).flatten()
    hdf5.create_dataset('admin/dates_index', data=np.asarray(hdf5_fred['admin/dates_index']))

    dset_agg = np.empty(shape=(601, repeat_arr.size),
                        dtype=np.float32)
    assert (dset_agg.shape == (601, np.asarray(hdf5_fred['zero_one']).size * 12))

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
        norm_dset = np.asarray(hdf5_fred[path])
        assert (norm_dset.shape[0] == 601)
        assert (norm_dset.dtype == np.float32)
        dset_agg[:, loc:norm_dset.shape[1]] = norm_dset[:, :]
        loc += norm_dset.shape[1]

    hdf5.create_dataset('data/train_x', data=dset_agg[0:597, :])
    hdf5.create_dataset('data/test_x', data=dset_agg[:-3, :])
    hdf5.close()
