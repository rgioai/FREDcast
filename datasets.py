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
    hdf5 = h5py.File('FREDcast.hdf5')

    gdp_dset = hdf5.create_dataset('admin/gdp', shape=(601, 1),
                                   dtype=np.float32)

    gdp = qd.get("FRED/GDP.1", returns='numpy', collapse='daily',
                 exclude_column_names=False, start_date='1967-4-1', end_date='2017-4-1')

    gdp.dtype.names = ('Date', 'Value')
    gdp['Value'] = gdp['Value'].astype(np.float32)
    gdp['Date'] = gdp['Date'].astype('datetime64[D]')
    gdp_values = time_scale(gdp['Value'], gdp['Date'])
    gdp_values = forward_fill(gdp_values)

    gdp_dset[:, 0] = gdp_values
    hdf5.close()


def gather_indicators(start, end, append=False):
    global START_TIME
    global TOTAL_CALLS
    global TOTAL_SLEEP
    global AUTH_TOKEN

    out = open('log.txt', 'w+')
    collection_timer = dt.datetime.now()

    # Create a backup of the old FREDcast.hdf5 as FREDcast.hdf5.bak
    # Load the incomplete dataset from FREDcast.hdf5 to memory
    # Assign the now free link to the last dset, and begin editing as normal

    if append is True:
        hdf5_old = h5py.File('FREDcast.hdf5')
        old_dset_raw = np.asarray(hdf5_old['data/raw'])
        old_dset_clean = np.asarray(hdf5_old['data/clean'])
        old_gdp = np.asarray(hdf5_old['admin/gdp'])
        old_index = np.asarray(hdf5_old['admin/values_index'])
        old_dates = np.asarray(hdf5_old['admin/dates_index'])
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
    indicators = np.asarray(data['Descriptions']).astype('S')

    hdf5 = h5py.File('FREDcast.hdf5')
    hdf5.require_group('data')
    hdf5.require_group('admin')

    if append is True:
        dset_raw = hdf5.create_dataset('data/raw', data=old_dset_raw)
        dset_clean = hdf5.create_dataset('data/clean', data=old_dset_clean)
        hdf5.create_dataset('admin/gdp', data=old_gdp)
        hdf5.create_dataset('admin/values_index', data=old_index)
        hdf5.create_dataset('admin/dates_index', data=old_dates)
        # Freeing space in memory
        del old_dset_raw
        del old_dset_clean
        del old_gdp
        del old_index
        del old_dates
    if append is False:
        gather_gdp()
        dset_raw = hdf5.create_dataset('data/raw', shape=(601, len(quandl_codes)),
                                       dtype=np.float32)
        dset_clean = hdf5.create_dataset('data/clean', shape=(601, len(quandl_codes)),
                                         dtype=np.float32)
        indicators = indicators[0:1000]
        hdf5.create_dataset('admin/values_index', data=indicators)
        date_list = []
        for i in range(0, 601, 1):
            date_list.append((np.datetime64('1967-04') + np.timedelta64(i, 'M')).astype(dt.datetime))
        dates = np.asarray(date_list).astype('S')
        hdf5.create_dataset('admin/dates_index', data=dates)

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
                TOTAL_CALLS += 1
            except qd.QuandlError as e:
                print(str(e) + '. IGNORING, retrying in 5 minutes.')
                sleep(300)
                TOTAL_SLEEP += 300
                pass
        quandl_values.dtype.names = ('Date', 'Value')
        quandl_values['Value'] = quandl_values['Value'].astype(np.float32)
        quandl_values['Date'] = quandl_values['Date'].astype('datetime64[D]')
        if quandl_values['Value'].size is not 0 and quandl_values['Date'].size is not 0:
            time_scaled = time_scale(quandl_values['Value'], quandl_values['Date'])
            assert (time_scaled.shape == (601,))
            dset_raw[:, i] = time_scaled
            forward_filled = forward_fill(time_scaled)
            assert (forward_filled.shape == (601,))
            dset_clean[:, i] = forward_filled
        else:
            dset_raw[:, i] = np.nan
            dset_clean[:, i] = np.nan

    out.write('Data collection runtime:' + str(dt.datetime.now() - collection_timer) + '\n')
    out.write('Total calls:' + str(TOTAL_CALLS) + '\n')
    out.write('Time spent waiting (in seconds):' + str(TOTAL_SLEEP) + '\n')

    if append is True:
        hdf5_old = h5py.File('FREDcast.hdf5.bak')

        filepaths = ['data/norm_data/zero_one',
                     'data/norm_data/zero_one_linear_residual',
                     'data/norm_data/zero_one_exp_residual',
                     'data/norm_data/zero_one_gdp_residual',
                     'data/norm_data/percent_change',
                     'data/norm_data/percent_change_linear_residual',
                     'data/norm_data/percent_change_exp_residual',
                     'data/norm_data/percent_change_gdp_residual',
                     'data/norm_data/normal_dist',
                     'data/norm_data/normal_dist_linear_residual',
                     'data/norm_data/normal_dist_exp_residual',
                     'data/norm_data/normal_dist_gdp_residual']

        for path in filepaths:
            n_dset = hdf5.create_dataset(path,
                                         data=np.asarray(hdf5_old[path]))
            normalizer_timer = dt.datetime.now()
            n_dset[:, start:end] = normalize_dataset(dset_clean[:, start:end], zero_one)
            out.write(path[15:] + ' runtime:' + str(dt.datetime.now() - normalizer_timer) + '\n')

        hdf5_old.close()

    if append is False:
        filepaths = ['data/norm_data/zero_one',
                     'data/norm_data/zero_one_linear_residual',
                     'data/norm_data/zero_one_exp_residual',
                     'data/norm_data/zero_one_gdp_residual',
                     'data/norm_data/percent_change',
                     'data/norm_data/percent_change_linear_residual',
                     'data/norm_data/percent_change_exp_residual',
                     'data/norm_data/percent_change_gdp_residual',
                     'data/norm_data/normal_dist',
                     'data/norm_data/normal_dist_linear_residual',
                     'data/norm_data/normal_dist_exp_residual',
                     'data/norm_data/normal_dist_gdp_residual']

        for path in filepaths:
            n_dset = hdf5.create_dataset(path, shape=(601, len(quandl_codes)),
                                         dtype=np.float32)
            normalizer_timer = dt.datetime.now()
            n_dset[:, start:end] = normalize_dataset(dset_clean[:, start:end], zero_one)
            out.write(path[15:] + ' runtime:' + str(dt.datetime.now() - normalizer_timer) + '\n')

    hdf5.close()

    out.write('Total runtime:' + str(dt.datetime.now() - START_TIME) + '\n')
    out.close()


if __name__ == '__main__':
    # gather_indicators(0, 1000, False)
    for i in range(12000, 300000, 1000):
        gather_indicators(i, i + 1000, True)
