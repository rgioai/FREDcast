import csv
from random import random
from time import sleep

import h5py
import quandl as qd

import settings
from cleaning_functions import *

s = settings.Settings()
s.load()
TOTAL_CALLS = 0
TOTAL_SLEEP = 0
AUTH_TOKEN = s.get('auth_code')


def gather_datasets():
    """
    Gathers various data from Quandl for usage in project.
    Writes data results to a hdf5 file.
    """

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

    def gather_indicators(start, end):
        global TOTAL_CALLS
        global TOTAL_SLEEP
        global AUTH_TOKEN

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

        # ISSUE: Can't recreate this after initial creation, error on following runs, will work around this ASAP
        dset = hdf5.create_dataset('data/sample_raw', shape=(601, len(quandl_codes)),
                                   dtype=np.float32)
        # In production, we'll probably use hdf5.require_dataset('data/raw'), after creating the empty dataset once.

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
                    sleepytime = random()
                    sleep(sleepytime)
                    TOTAL_SLEEP += sleepytime
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

        hdf5.close()

    # gather_gdp()
    gather_indicators(s.get('start'), s.get('end'))


if __name__ == '__main__':
    gather_datasets()
