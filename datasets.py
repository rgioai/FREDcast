from random import randint
from time import sleep

import h5py
import numpy as np
import quandl as qd

import settings

TOTAL_CALLS = 0
AUTH_TOKEN = "W1THAw2psKF66k1Px9tz"


def gather_datasets():
    """
    Gathers various data from Quandl for usage in project.
    Writes data results to a hdf5 file.
    """

    def gather_y():
        s = settings.Settings()
        s.load()
        global TOTAL_CALLS
        global AUTH_TOKEN
        calls = []
        for value in ['unemployment', 'payroll', 'gdp', 'cpi']:
            calls.append(s.get(value))
        values = []
        dates = []
        targets = []
        for call, target_type in zip(calls, ['Target Unemployment', 'Target Payroll', 'Target GDP', 'Target CPI']):
            TOTAL_CALLS += 1
            print(call, TOTAL_CALLS)
            returned_call = None
            while returned_call is None:
                try:
                    returned_call = qd.get(call + ".1", returns='numpy', collapse='monthly',
                                           exclude_column_names=False, start_date='1967-3-1', end_date='2017-3-1',
                                           authtoken=AUTH_TOKEN)
                    returned_call.dtype.names = ('Date', 'Value')
                    value = returned_call['Value'].astype(np.float32)
                    date = returned_call['Date']
                    target = target_type
                except qd.QuandlError:
                    sleep(60)
                    pass
            sleep(randint(1, 3))
            values.append(value)
            dates.append(date)
            targets.append(target)

        return [values, dates, targets]

    def gather_x(start, stop):
        s = settings.Settings()
        s.load()
        global TOTAL_CALLS
        global AUTH_TOKEN
        values = []
        dates = []
        features = []
        calls = s.get('features')
        if stop > len(calls):
            stop = len(calls)
        calls = calls[start:stop + 1]
        for call in calls:
            TOTAL_CALLS += 1
            print(call[0], TOTAL_CALLS)
            if TOTAL_CALLS % 2000 == 0:
                sleep(600)
            returned_call = None
            while returned_call is None:
                try:
                    returned_call = qd.get(call[0] + ".1", returns='numpy', collapse='monthly',
                                           exclude_column_names=False, start_date='1967-3-1', end_date='2017-3-1',
                                           authtoken=AUTH_TOKEN)
                    returned_call.dtype.names = ('Date', 'Value')
                    value = returned_call['Value'].astype(np.float32)
                    date = returned_call['Date']
                    feature = call[1]
                except qd.QuandlError:
                    sleep(60)
                    pass
            sleep(randint(1, 3))
            values.append(value)
            dates.append(date)
            features.append(feature)

        return [values, dates, features]

    s = settings.Settings()
    s.load()

    y = gather_y()
    x = gather_x(0, 10000)

    f = h5py.File('data.h5', 'w')
    f.create_dataset('value', data=y)
    f.close()

    for i in range(10000, len(settings.get('features')), 10000):
        x = gather_x(i, i + 10000)
        f = h5py.File('data.h5', 'a')
        f.create_dataset('value', data=x)
        f.close()


if __name__ == '__main__':
    gather_datasets()
