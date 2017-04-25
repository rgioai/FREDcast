import numpy as np
import h5py


def group_name(norm_fn, residual_fn):
    if norm_fn not in ['normal_dist', 'percent_change', 'zero_one', '']:
        raise ValueError('Invalid norm function')
    if residual_fn not in ['exp_residual', 'gdp_residual', 'linear_residual', 'none', '']:
        raise ValueError('Invalid residual function')
    if norm_fn == '':
        norm_fn = 'zero_one'
    if residual_fn == 'none':
        residual_fn = ''
    if residual_fn != '':
        return norm_fn + '_' + residual_fn
    else:
        return norm_fn


def get_hdf5(aggregate, sample, path='/centurion/FREDcast/'):
    if aggregate:
        hdf5_name = 'rnn_data'
    else:
        hdf5_name = 'split_data'
    if sample:
        hdf5_name += '_sample'
    return path + hdf5_name + '.hdf5'


def y_index(indicator):
    try:
        return ['gdp', 'cpi', 'payroll', 'unemployment'].index(indicator)
    except ValueError:
        raise ValueError('%s not a valid indicator' % indicator)


def get_data(data_tuple, dsets=False,
             norm_fn='', residual_fn='', aggregate=False, sample=True,
             indicator='gdp'):
    if data_tuple is not None:
        indicator = data_tuple[0]
        norm_fn = data_tuple[1]
        residual_fn = data_tuple[2]
        aggregate = data_tuple[3]
        sample = data_tuple[4]


    hdf5 = h5py.File(get_hdf5(aggregate, sample), 'r')
    grp = hdf5[group_name(norm_fn, residual_fn)]

    x_train, x_test, y_train, y_test = grp['train_x'], grp['test_x'], grp['train_y'], grp['test_y']

    y_train = y_train[:, y_index(indicator)]
    y_test = y_test[:, y_index(indicator)]

    if dsets:  # Return the h5py dset objects
        return x_train, x_test, y_train, y_test
    else:  # Return the actual data
        return x_train[:], x_test[:], y_train[:], y_test[:]
