import sys
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
        types = ['unemployment', 'payroll', 'gdp', 'cpi']
        for type in types:
            calls.append(s.get(type))
        file_values = h5py.File('y_values.h5', 'w')
        file_dates = h5py.File('y_dates.h5', 'w')
        for call, type in zip(calls, types):
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

            file_values.create_dataset(type, data=value)
            file_dates.create_dataset(type, data=date)
        file_dates.close()
        file_values.close()

    def gather_x(start, stop, pos):
        global TOTAL_CALLS
        global AUTH_TOKEN
        calls = s.get('features')
        if stop > len(calls):
            stop = len(calls)
        calls = calls[start - 1:stop]
        file_values = h5py.File('x_values.h5', 'a')
        file_dates = h5py.File('x_dates.h5', 'a')
        for call in calls:
            TOTAL_CALLS += 1
            print(call[0], pos)
            pos += 1
            if TOTAL_CALLS % 2000 == 0:
                sleep(120)
            returned_call = None
            while returned_call is None:
                try:
                    returned_call = qd.get(call[0] + ".1", returns='numpy', collapse='monthly',
                                           exclude_column_names=False, start_date='1967-3-1', end_date='2017-3-1',
                                           authtoken=AUTH_TOKEN)
                    returned_call.dtype.names = ('Date', 'Value')
                    value = returned_call['Value'].astype(np.float32)
                    date = returned_call['Date'].astype('S10')
                except qd.QuandlError as e:
                    print(str(e) + 'Ignoring, retrying in 1 minute.')
                    sleep(60)
                    pass
            sleep(random())
            file_values.create_dataset(call[0], data=value)
            file_dates.create_dataset(call[0], data=date)
        file_values.close()
        file_dates.close()

    gather_y()
    gather_x(s.get('start'), s.get('end'), s.get('start'))

    #len(s.get('features'))

if __name__ == '__main__':
    gather_datasets()
