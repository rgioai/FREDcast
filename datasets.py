import csv
import os
from time import sleep

import quandl as qd

import settings
from cleaning_functions import *
from norm_functions import *

s = settings.Settings()
s.load()
START_TIME = dt.datetime.now()
TOTAL_CALLS = 0
TOTAL_SLEEP = 0
AUTH_TOKEN = s.get('auth_code')


def gather_gdp():
    hdf5 = h5py.File('GDP.hdf5')
    hdf5.require_group('data')

    dset = hdf5.create_dataset('data/gdp', shape=(601, 1),
                               dtype=np.float32)

    gdp = qd.get("FRED/GDP.1", returns='numpy', collapse='daily',
                 exclude_column_names=False, start_date='1967-4-1', end_date='2017-4-1')

    gdp.dtype.names = ('Date', 'Value')
    gdp['Value'] = gdp['Value'].astype(np.float32)
    gdp['Date'] = gdp['Date'].astype('datetime64[D]')
    gdp_values = time_scale(gdp['Value'], gdp['Date'])
    gdp_values = forward_fill(gdp_values)

    hdf5 = h5py.File('GDP.hdf5')
    dset[:, 0] = gdp_values
    hdf5.close()


def gather_indicators(start, end, append=False):
    global START_TIME
    global TOTAL_CALLS
    global TOTAL_SLEEP
    global AUTH_TOKEN

    # ISSUE: Datasets are hard to edit once they're already set.
    # The current workaround is:
    # Create a backup of the old FREDcast.hdf5 as FREDcast.hdf5.bak
    # Load the incomplete dataset from FREDcast.hdf5 to memory
    # Assign the now free link to the last dset, and begin editing as normal

    if append is True:
        hdf5_old = h5py.File('FREDcast.hdf5')
        old_dset = np.asarray(hdf5_old['data/raw'])

        old_norm_fns = [np.asarray(hdf5_old['data/norm_data/linear_residual']),
                        np.asarray(hdf5_old['data/norm_data/linear_residual_zero_one']),
                        np.asarray(hdf5_old['data/norm_data/linear_residual_percent_change']),
                        np.asarray(hdf5_old['data/norm_data/linear_residual_normal_dist']),
                        np.asarray(hdf5_old['data/norm_data/exp_residual']),
                        np.asarray(hdf5_old['data/norm_data/exp_residual_zero_one']),
                        np.asarray(hdf5_old['data/norm_data/exp_residual_percent_change']),
                        np.asarray(hdf5_old['data/norm_data/exp_residual_normal_dist']),
                        np.asarray(hdf5_old['data/norm_data/gdp_residual']),
                        np.asarray(hdf5_old['data/norm_data/gdp_residual_zero_one']),
                        np.asarray(hdf5_old['data/norm_data/gdp_residual_percent_change']),
                        np.asarray(hdf5_old['data/norm_data/gdp_residual_normal_dist'])]

        hdf5_old.close()
        os.rename(os.path.realpath('FREDcast.hdf5'), os.path.realpath('FREDcast.hdf5') + '.bak')

    pos = start

    with open('quandl_codes.csv', 'r') as f:
        reader = csv.DictReader(f)
        data = {}
        for row in reader:
            for header, value in row.items():
                try:
                    data[header].append(value)
                except KeyError:
                    data[header] = [value]

    quandl_codes = data['Codes']

    hdf5 = h5py.File('FREDcast.hdf5')
    hdf5.require_group('data')

    if append is True:
        dset = hdf5.create_dataset('data/raw', data=old_dset)
    if append is False:
        dset = hdf5.create_dataset('data/raw', shape=(601, len(quandl_codes)),
                                   dtype=np.float32)

    if start > len(quandl_codes):
        start = len(quandl_codes) - 1
    if end > len(quandl_codes):
        end = len(quandl_codes)
    for i in range(start, end):
        pos += 1
        print(quandl_codes[i], pos)

        quandl_code = quandl_codes[i]
        quandl_values = None
        while quandl_values is None:
            try:
                quandl_values = qd.get(quandl_code + ".1", returns='numpy', collapse='daily',
                                       exclude_column_names=False, start_date='1967-4-1', end_date='2017-4-1',
                                       auth_token=AUTH_TOKEN)
                sleep(0.1)
                TOTAL_SLEEP += 0.1
                TOTAL_CALLS += 1
            except qd.QuandlError as e:
                print(str(e) + '. IGNORING, retrying in 1 minute.')
                sleep(60)
                TOTAL_SLEEP += 60
                pass
        quandl_values.dtype.names = ('Date', 'Value')
        quandl_values['Value'] = quandl_values['Value'].astype(np.float32)
        quandl_values['Date'] = quandl_values['Date'].astype('datetime64[D]')
        time_scaled = time_scale(quandl_values['Value'], quandl_values['Date'])
        forward_filled = forward_fill(time_scaled)
        assert (forward_filled.shape == (601,))
        dset[:, i] = forward_filled

    if append is True:
        new_data = normalize_dataset(dset[:, start:end], linear_residual)
        hdf5.create_dataset('data/norm_data/linear_residual',
                            data=np.concatenate((old_norm_fns[0], new_data)))

        n_dset1 = hdf5.create_dataset('data/norm_data/linear_residual', data=old_norm_fns[0])
        n_dset1[:, start:end] = normalize_dataset(dset[:, start:end], linear_residual)

        n_dset2 = hdf5.create_dataset('data/norm_data/linear_residual_zero_one', data=old_norm_fns[1])
        n_dset2[:, start:end] = normalize_dataset(dset[:, start:end], linear_residual, zero_one)

        n_dset3 = hdf5.create_dataset('data/norm_data/linear_residual_percent_change', data=old_norm_fns[2])
        n_dset3[:, start:end] = normalize_dataset(dset[:, start:end], linear_residual, percent_change)

        n_dset4 = hdf5.create_dataset('data/norm_data/linear_residual_normal_dist', data=old_norm_fns[3])
        n_dset4[:, start:end] = normalize_dataset(dset[:, start:end], linear_residual, normal_dist)

        n_dset5 = hdf5.create_dataset('data/norm_data/exp_residual', data=old_norm_fns[4])
        n_dset5[:, start:end] = normalize_dataset(dset[:, start:end], exp_residual)

        n_dset6 = hdf5.create_dataset('data/norm_data/exp_residual_zero_one', data=old_norm_fns[5])
        n_dset6[:, start:end] = normalize_dataset(dset[:, start:end], exp_residual, zero_one)

        n_dset7 = hdf5.create_dataset('data/norm_data/exp_residual_percent_change', data=old_norm_fns[6])
        n_dset7[:, start:end] = normalize_dataset(dset[:, start:end], exp_residual, percent_change)

        n_dset8 = hdf5.create_dataset('data/norm_data/exp_residual_normal_dist', data=old_norm_fns[7])
        n_dset8[:, start:end] = normalize_dataset(dset[:, start:end], exp_residual, normal_dist)

        n_dset9 = hdf5.create_dataset('data/norm_data/gdp_residual', data=old_norm_fns[8])
        n_dset9[:, start:end] = normalize_dataset(dset[:, start:end], gdp_residual)

        n_dset10 = hdf5.create_dataset('data/norm_data/gdp_residual_zero_one', data=old_norm_fns[9])
        n_dset10[:, start:end] = normalize_dataset(dset[:, start:end], gdp_residual, zero_one)

        n_dset11 = hdf5.create_dataset('data/norm_data/gdp_residual_percent_change', data=old_norm_fns[10])
        n_dset11[:, start:end] = normalize_dataset(dset[:, start:end], gdp_residual, percent_change)

        n_dset12 = hdf5.create_dataset('data/norm_data/gdp_residual_normal_dist', data=old_norm_fns[11])
        n_dset12[:, start:end] = normalize_dataset(dset[:, start:end], gdp_residual, normal_dist)

    if append is False:
        n_dset1 = hdf5.create_dataset('data/norm_data/linear_residual', shape=(601, len(quandl_codes)),
                                   dtype=np.float32)
        n_dset1[:, start:end] = normalize_dataset(dset[:, start:end], linear_residual)

        n_dset2 = hdf5.create_dataset('data/norm_data/linear_residual_zero_one', shape=(601, len(quandl_codes)),
                                   dtype=np.float32)
        n_dset2[:, start:end] = normalize_dataset(dset[:, start:end], linear_residual, zero_one)

        n_dset3 = hdf5.create_dataset('data/norm_data/linear_residual_percent_change', shape=(601, len(quandl_codes)),
                                   dtype=np.float32)
        n_dset3[:, start:end] = normalize_dataset(dset[:, start:end], linear_residual, percent_change)

        n_dset4 = hdf5.create_dataset('data/norm_data/linear_residual_normal_dist', shape=(601, len(quandl_codes)),
                                   dtype=np.float32)
        n_dset4[:, start:end] = normalize_dataset(dset[:, start:end], linear_residual, normal_dist)

        n_dset5 = hdf5.create_dataset('data/norm_data/exp_residual', shape=(601, len(quandl_codes)),
                                    dtype=np.float32)
        n_dset5[:, start:end] = normalize_dataset(dset[:, start:end], exp_residual)

        n_dset6 = hdf5.create_dataset('data/norm_data/exp_residual_zero_one', shape=(601, len(quandl_codes)),
                                    dtype=np.float32)
        n_dset6[:, start:end] = normalize_dataset(dset[:, start:end], exp_residual, zero_one)

        n_dset7 = hdf5.create_dataset('data/norm_data/exp_residual_percent_change', shape=(601, len(quandl_codes)),
                                    dtype=np.float32)
        n_dset7[:, start:end] = normalize_dataset(dset[:, start:end], exp_residual, percent_change)

        n_dset8 = hdf5.create_dataset('data/norm_data/exp_residual_normal_dist', shape=(601, len(quandl_codes)),
                                    dtype=np.float32)
        n_dset8[:, start:end] = normalize_dataset(dset[:, start:end], exp_residual, normal_dist)

        n_dset9 = hdf5.create_dataset('data/norm_data/gdp_residual', shape=(601, len(quandl_codes)),
                                    dtype=np.float32)
        n_dset9[:, start:end] = normalize_dataset(dset[:, start:end], gdp_residual)

        n_dset10 = hdf5.create_dataset('data/norm_data/gdp_residual_zero_one', shape=(601, len(quandl_codes)),
                                    dtype=np.float32)
        n_dset10[:, start:end] = normalize_dataset(dset[:, start:end], gdp_residual, zero_one)

        n_dset11 = hdf5.create_dataset('data/norm_data/gdp_residual_percent_change', shape=(601, len(quandl_codes)),
                                    dtype=np.float32)
        n_dset11[:, start:end] = normalize_dataset(dset[:, start:end], gdp_residual, percent_change)

        n_dset12 = hdf5.create_dataset('data/norm_data/gdp_residual_normal_dist', shape=(601, len(quandl_codes)),
                                    dtype=np.float32)
        n_dset12[:, start:end] = normalize_dataset(dset[:, start:end], gdp_residual, normal_dist)

    hdf5.close()
    print('Total runtime:' + str(dt.datetime.now() - START_TIME))
    print('Total calls:' + str(TOTAL_CALLS))
    print('Time spent waiting (in seconds):' + str(TOTAL_SLEEP))


if __name__ == '__main__':
    gather_indicators(0, 10, False)
    gather_indicators(10, 20, True)
