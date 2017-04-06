from random import random
from time import sleep

import h5py
import numpy as np
import quandl as qd

import settings

s = settings.Settings()
s.load()
TOTAL_CALLS = 0
AUTH_TOKEN = s.get('auth_code')


def gather_datasets():
    """
    Gathers various data from Quandl for usage in project.
    Writes data results to a hdf5 file.
    """

    def gather_y():
        global TOTAL_CALLS
        global AUTH_TOKEN
        calls = []
        for value in ['unemployment', 'payroll', 'gdp', 'cpi']:
            calls.append(s.get(value))
        values = np.empty(shape=(1,), dtype=np.float32)
        dates = np.empty(shape=(1,), dtype='S10')
        for call in calls:
            TOTAL_CALLS += 1
            print(call)
            returned_call = None
            while returned_call is None:
                try:
                    returned_call = qd.get(call + ".1", returns='numpy', collapse='monthly',
                                           exclude_column_names=False, start_date='1967-3-1', end_date='2017-3-1',
                                           authtoken=AUTH_TOKEN)
                    returned_call.dtype.names = ('Date', 'Value')
                    value = returned_call['Value'].astype(np.float32)
                    date = returned_call['Date'].astype('S10')
                except qd.QuandlError:
                    sleep(60)
                    pass
            sleep(random())
            np.append(values, value)
            np.append(dates, date)

        return [values, dates]

    def gather_x(start, stop, pos):
        global TOTAL_CALLS
        global AUTH_TOKEN
        values = np.empty(shape=(1,), dtype=np.float32)
        dates = np.empty(shape=(1,), dtype='S10')
        features = np.empty(shape=(1,), dtype='S10')
        calls = s.get('features')
        if stop > len(calls):
            stop = len(calls)
        calls = calls[start:stop]
        for call in calls:
            TOTAL_CALLS += 1
            pos += 1
            print(call[0], pos)
            if TOTAL_CALLS % 2000 == 0:
                sleep(300)
            returned_call = None
            while returned_call is None:
                try:
                    returned_call = qd.get(call[0] + ".1", returns='numpy', collapse='monthly',
                                           exclude_column_names=False, start_date='1967-3-1', end_date='2017-3-1',
                                           authtoken=AUTH_TOKEN)
                    returned_call.dtype.names = ('Date', 'Value')
                    value = returned_call['Value'].astype(np.float32)
                    date = returned_call['Date'].astype('S10')
                    feature = np.string_(call[1])
                except qd.QuandlError as e:
                    print(str(e) + 'Ignoring, retrying in 1 minute.')
                    sleep(60)
                    pass
            sleep(random())
            np.append(values, value)
            np.append(dates, date)
            np.append(features, feature)

        return [values, dates, features]

    y = gather_y()

    f = h5py.File('y_values.h5', 'w')
    f.create_dataset('values_predict', data=y[0])
    f.close()

    f = h5py.File('y_dates.h5', 'w')
    f.create_dataset('dates_predict', data=y[1])
    f.close()

    for i in range(s.get('jump') * 1000, len(s.get('features')), 1000):
        x = gather_x(i, i + 1001, i)
        f = h5py.File('x_values.h5', 'a')
        f.create_dataset('values' + str(i + 1) + 'to' + str(i + 1000), data=x[0])
        f.close()

        f = h5py.File('x_dates.h5', 'a')
        f.create_dataset('dates' + str(i + 1) + 'to' + str(i + 1000), data=x[1])
        f.close()

        f = h5py.File('x_features.h5', 'a')
        f.create_dataset('features' + str(i + 1) + 'to' + str(i + 1000), data=x[2])
        f.close()


if __name__ == '__main__':
    gather_datasets()
