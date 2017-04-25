from time import sleep

import quandl as qd

import settings
from cleaning_functions import *
from norm_functions import *

s = settings.Settings()
s.load()
TOTAL_CALLS = 0
TOTAL_SLEEP = 0
AUTH_CODES = s.get('auth_codes')
CURRENT_AUTH = 0


def gather_y(sample):
    if sample is True:
        hdf5 = h5py.File('FREDcast_sample.hdf5')
    else:
        hdf5 = h5py.File('FREDcast.hdf5')

    dsets = [hdf5.create_dataset('admin/gdp', shape=(601, 1),
                                 dtype=np.float32),
             hdf5.create_dataset('admin/cpi', shape=(601, 1),
                                 dtype=np.float32),
             hdf5.create_dataset('admin/payroll', shape=(601, 1),
                                 dtype=np.float32),
             hdf5.create_dataset('admin/unemployment', shape=(601, 1),
                                 dtype=np.float32)]

    codes = ['FRED/GDP.1', 'FRED/CPIAUCSL.1', 'FRED/PAYEMS.1', 'FRED/UNEMPLOY.1']

    for dset, code in zip(dsets, codes):
        quandl_values = qd.get(code, returns='numpy', collapse='daily',
                               exclude_column_names=False, start_date='1967-4-1', end_date='2017-4-1',
                               auth_token='FM7y_rayzsW5AXASS_kD')

        quandl_values.dtype.names = ('Date', 'Value')
        quandl_values['Value'] = quandl_values['Value'].astype(np.float32)
        quandl_values['Date'] = quandl_values['Date'].astype('datetime64[D]')
        time_scaled = time_scale(quandl_values['Value'], quandl_values['Date'])
        forward_filled = forward_fill(time_scaled)

        dset[:, 0] = forward_filled

    hdf5.close()


def create_admin_hdf5(sample=False):
    with open('quandl_codes.csv', 'r') as f:
        reader = csv.DictReader(f)
        data = {}
        for row in reader:
            for header, value in row.items():
                try:
                    data[header].append(value)
                except KeyError:
                    data[header] = [value]

    if sample is True:
        hdf5 = h5py.File('admin_sample.hdf5')
        codes = np.asarray(data['Codes']).astype('S')
        descriptions = np.asarray(data['Descriptions']).astype('S')
        codes = codes[0:1000]
        descriptions = descriptions[0:1000]

        hdf5.create_dataset('admin/codes', data=codes)
        hdf5.create_dataset('admin/descriptions', data=descriptions)
        hdf5.close()

    else:
        hdf5 = h5py.File('admin.hdf5')
        codes = np.asarray(data['Codes']).astype('S')
        descriptions = np.asarray(data['Descriptions']).astype('S')

        hdf5.create_dataset('admin/codes', data=codes)
        hdf5.create_dataset('admin/descriptions', data=descriptions)
        hdf5.close()


def gather_data(start, end, append=False, sample=False):
    global TOTAL_CALLS
    global TOTAL_SLEEP
    global AUTH_CODES
    global CURRENT_AUTH

    out = open('collection_log.txt', 'w+')
    collection_timer = dt.datetime.now()

    # Create a backup of the old FREDcast.hdf5 as FREDcast.hdf5.bak
    # Load the incomplete dataset from FREDcast.hdf5 to memory
    # Assign the now free link to the last dset, and begin editing as normal
    if sample is True:
        start = 0
        end = 1000
        append = False

    if append is True:
        hdf5_old = h5py.File('FREDcast.hdf5')
        old_dset_raw = np.asarray(hdf5_old['data/raw'])
        old_dset_clean = np.asarray(hdf5_old['data/clean'])
        old_gdp = np.asarray(hdf5_old['admin/gdp'])
        old_cpi = np.asarray(hdf5_old['admin/cpi'])
        old_payroll = np.asarray(hdf5_old['admin/payroll'])
        old_unemployment = np.asarray(hdf5_old['admin/unemployment'])
        old_dates = np.asarray(hdf5_old['admin/dates_index'])
        hdf5_old.close()
        os.rename(os.path.realpath('FREDcast.hdf5'), os.path.realpath('FREDcast.hdf5') + '.bak')

    pos = start

    if sample is True:
        hdf5_admin = h5py.File('admin_sample.hdf5')
        hdf5 = h5py.File('FREDcast_sample.hdf5')
        hdf5.require_group('data')
        hdf5.require_group('admin')
    else:
        hdf5_admin = h5py.File('admin.hdf5')
        hdf5 = h5py.File('FREDcast.hdf5')
        hdf5.require_group('data')
        hdf5.require_group('admin')

    quandl_codes = np.asarray(hdf5_admin['admin/codes'])
    hdf5_admin.close()

    if append is True:
        dset_raw = hdf5.create_dataset('data/raw', data=old_dset_raw)
        dset_clean = hdf5.create_dataset('data/clean', data=old_dset_clean)
        hdf5.create_dataset('admin/gdp', data=old_gdp)
        hdf5.create_dataset('admin/cpi', data=old_cpi)
        hdf5.create_dataset('admin/payroll', data=old_payroll)
        hdf5.create_dataset('admin/unemployment', data=old_unemployment)
        hdf5.create_dataset('admin/dates_index', data=old_dates)
        # Freeing space in memory
        del old_dset_raw
        del old_dset_clean
        del old_gdp
        del old_cpi
        del old_payroll
        del old_unemployment
        del old_dates
    if append is False:
        gather_y(sample)
        dset_raw = hdf5.create_dataset('data/raw', shape=(601, len(quandl_codes)),
                                       dtype=np.float32)
        dset_clean = hdf5.create_dataset('data/clean', shape=(601, len(quandl_codes)),
                                         dtype=np.float32)
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
        quandl_code = str(quandl_codes[i])[2:-1]
        print(quandl_code, pos)
        quandl_values = None
        while quandl_values is None:
            try:
                quandl_values = qd.get(quandl_code + ".1", returns='numpy', collapse='daily',
                                       exclude_column_names=False, start_date='1967-4-1', end_date='2017-4-1',
                                       auth_token=AUTH_CODES[CURRENT_AUTH])
                TOTAL_CALLS += 1
                looping = False
            except qd.QuandlError as e:
                if "daily" in str(e):
                    print('Encountered max limit. Switching API key.')
                    CURRENT_AUTH += 1
                    if CURRENT_AUTH > len(AUTH_CODES) - 1:
                        looping = True
                        CURRENT_AUTH = 0
                    if CURRENT_AUTH > len(AUTH_CODES) - 1 and looping:
                        print('Expended all API keys. Terminating script, add more API keys in the future.')
                        quit()
                else:
                    print(str(e) + '. IGNORING, retrying in 5 minutes.')
                    sleep(300)
                    TOTAL_SLEEP += 300
                    pass
        quandl_values.dtype.names = ('Date', 'Value')
        quandl_values['Value'] = quandl_values['Value'].astype(np.float32)
        quandl_values['Date'] = quandl_values['Date'].astype('datetime64[D]')
        if quandl_values['Value'].size > 1 and quandl_values['Date'].size > 1:
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

    hdf5.close()


def normalize_data(sample=False):
    if sample is True:
        hdf5 = h5py.File('FREDcast_sample.hdf5')
    else:
        hdf5 = h5py.File('FREDcast.hdf5')

    out = open('normalization_log.txt', 'w+')

    dset_clean = np.asarray(hdf5['data/clean'])

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

    functions = [[zero_one, None],
                 [linear_residual, zero_one],
                 [exp_residual, zero_one],
                 [gdp_residual, zero_one],
                 [percent_change, None],
                 [linear_residual, percent_change],
                 [exp_residual, percent_change],
                 [gdp_residual, percent_change],
                 [normal_dist, None],
                 [linear_residual, normal_dist],
                 [exp_residual, normal_dist],
                 [gdp_residual, normal_dist]]

    for path, function in zip(filepaths, functions):
        print(path[15:])
        n_dset = hdf5.create_dataset(path, shape=(dset_clean.shape[0], dset_clean.shape[1]),
                                     dtype=np.float32)
        normalizer_timer = dt.datetime.now()
        n_dset[:, :] = normalize_dataset(dset_clean[:, :], function[0], function[1])
        out.write(path[15:] + ' runtime:' + str(dt.datetime.now() - normalizer_timer) + '\n')

    out.close()
    hdf5.close()


def modify_data(sample=False):
    if sample is True:
        truncate_hdf5('FREDcast_sample.hdf5', 328)
        remove_nan_features('FREDcast_sample.hdf5', 'admin_sample.hdf5')
    else:
        truncate_hdf5('FREDcast.hdf5', 328)
        remove_nan_features('FREDcast.hdf5', 'admin.hdf5')
    normalize_data(sample=sample)


if __name__ == '__main__':
    # FUTURE: I will move all this to the interface.

    # create_admin_hdf5(sample=False)
    # create_admin_hdf5(sample=True)

    # print('Sample Data')
    # modify_data(sample=True)
    # print('Full Data')
    # modify_data(sample=False)

    hdf5 = h5py.File('FREDcast.hdf5')
    raw = np.asarray(hdf5['data/raw'])
    clean = np.asarray(hdf5['data/clean'])
    truncate_loss(clean)
    forward_fill_loss(raw, clean)
    hdf5.close()

    # gather_data(0, 1000, False, sample=True)

    # gather_data(0, 1000, False)
    # start = s.get('start')
    # for i in range(start, 339000, 1000):
    #     gather_data(i, i + 1000, True)
    # gather_data(339000, 339870, True)
