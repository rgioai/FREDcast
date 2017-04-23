import csv
import datetime as dt
import warnings

import numpy as np


def forward_fill(data_column):
    copy = np.empty_like(data_column)
    copy[:] = data_column
    copy[np.isnan(copy)] = 0

    prev = np.arange(len(copy))
    prev[copy == 0] = 0
    prev = np.maximum.accumulate(prev)
    filled = copy[prev]
    filled[filled == 0] = np.nan
    return filled


def time_scale(data_column, date_column, freq=None):
    # Removed redundancies, as they still provided accurate calculations even with different 'intervals'.

    if freq == 'monthly':
        scaled_data = np.empty(shape=(601,), dtype=np.float32)
        scaled_data[:] = np.nan

        start_date = np.datetime64('1967-04')
        date_list = []
        for i in range(0, 601, 1):
            date_list.append((start_date + np.timedelta64(i, 'M')).astype(dt.datetime))

        date_list = np.asarray(date_list)

        indices_dl = np.arange(date_list.shape[0])[np.in1d(date_list, date_column)]
        indices_dc = np.arange(date_column.shape[0])[np.in1d(date_column, date_list)]

        for i in range(0, len(indices_dl), 1):
            scaled_data[indices_dl[i]] = data_column[indices_dc[i]]

        return trim_column(scaled_data)

    elif freq == 'daily':
        scaled_data = np.empty(shape=(18264,), dtype=np.float32)
        scaled_data[:] = np.nan

        start_date = np.datetime64('1967-04-01')
        date_list = []
        for i in range(0, 18264, 1):
            date_list.append((start_date + np.timedelta64(i, 'D')).astype(dt.datetime))

        date_list = np.asarray(date_list)

        indices_dl = np.arange(date_list.shape[0])[np.in1d(date_list, date_column)]
        indices_dc = np.arange(date_column.shape[0])[np.in1d(date_column, date_list)]

        for i in range(0, len(indices_dl), 1):
            scaled_data[indices_dl[i]] = data_column[indices_dc[i]]

        start, stop = 0, 0
        for i in range(0, 601, 1):
            month = np.arange(
                np.datetime64('1967-04') + np.timedelta64(i, 'M'), (np.datetime64('1967-05') + np.timedelta64(i, 'M')),
                dtype='datetime64[D]')

            start = stop
            stop += month.shape[0]

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                scaled_data[i] = np.nanmean(scaled_data[start:stop])

        return trim_column(scaled_data)

    elif freq is None:
        # Added some wriggle room for business days
        first_date, second_date = date_column[0], date_column[1]
        daydiff = (second_date - first_date).days
        if 1 <= daydiff < 28:
            return time_scale(data_column, date_column, 'daily')
        elif 28 <= daydiff:
            return time_scale(data_column, date_column, 'monthly')
        # if 60 <= daydiff < 340:
        #    return time_scale(data_column, date_column, 'quarterly')
        # elif daydiff <= 340:
        #    return time_scale(data_column, date_column, 'annual')
        else:
            raise ValueError('Negative date difference')
    else:
        raise ValueError('Frequency value not recognized')


def trim_column(data_column, max_length=601):
    """
    Abstraction of numpy slicing operation for single column truncation.
    :param data_column: 1D np.array
    :param max_length: int of truncation index
    :return: np.array
    """
    assert (len(data_column.shape) == 1)  # Assert data_column is a 1D numpy array
    return data_column[:max_length]


def truncate_loss(dataset, isTest=False):
    """
    Returns percentage of columns with np.nan values; this indicates what percentage of features do not extend to the
    index specified.  (All other np.nan values should be forward filled)
    :param dataset: 2D np.array
    :param isTest: bool to prevent overwriting normal outfile
    """
    date_list = []
    percent_list = []
    for i in range(0, 601, 1):
        date_list.append((np.datetime64('1967-04') + np.timedelta64(i, 'M')).astype(dt.datetime))
        percent_list.append(np.count_nonzero(np.isnan(dataset[i, :])) / dataset.shape[1])

    filename = 'truncate_loss.csv'
    if isTest:
        filename = 'truncate_loss_UT.csv'

    with open(filename, 'w+') as csvfile:
        for date, percent in zip(date_list, percent_list):
            csvfile.write(str(date) + ", " + str(percent * 100) + "%")
            csvfile.write('\n')


def forward_fill_loss(dataset_raw, dataset_clean, isTest=False):
    """
    Returns percentage of columns that had forward-filled values.
    :param dataset_clean: 2D np.array
    :param dataset_raw: 2D np.array without ffill
    :param isTest: bool to prevent overwriting normal outfile
    """
    date_list = []
    percent_list = []
    # month
    for i in range(0, dataset_raw.shape[0], 1):
        percent_nan = 0
        date_list.append((np.datetime64('1967-04') + np.timedelta64(i, 'M')).astype(dt.datetime))
        nan_indicies = np.argwhere(np.isnan(dataset_raw[i, :]))
        for index in nan_indicies:
            if dataset_clean[i, index] is not np.nan:
                percent_nan += 1
        percent_list.append(percent_nan / dataset_raw.shape[1])

    filename1 = 'ffill_loss_month.csv'
    if isTest:
        filename1 = 'ffill_loss_month_UT.csv'

    with open(filename1, 'w+') as csvfile:
        for date, percent in zip(date_list, percent_list):
            csvfile.write(str(date) + ", " + str(percent * 100) + "%")
            csvfile.write('\n')

    feature_list = range(1, dataset_raw.shape[1] + 1)
    percent_list = []
    # feature
    for i in range(0, dataset_raw.shape[1], 1):
        percent_nan = 0
        nan_indicies = np.argwhere(np.isnan(dataset_raw[:, i]))
        for index in nan_indicies:
            if dataset_clean[index, i] is not np.nan:
                percent_nan += 1
        percent_list.append(percent_nan / 601)

    filename1 = 'ffill_loss_feature.csv'
    if isTest:
        filename1 = 'ffill_loss_feature_UT.csv'

    with open(filename1, 'w+') as csvfile:
        for feature, percent in zip(feature_list, percent_list):
            csvfile.write(str(feature) + ", " + str(percent * 100) + "%")
            csvfile.write('\n')


def truncate_column(data_column, truncation_point):
    """
    Abstraction of numpy slicing operation for 2D array truncation.
    :param dataset: 2D np.array
    :param truncation_point: int of truncation index to test
    :return: np.array
    """
    assert (len(data_column.shape) == 1)  # Assert data_column is a 1D numpy array
    return data_column[-1 * truncation_point:]


def truncate_dataset(dataset, truncation_point):
    """
    Abstraction of numpy slicing operation for 2D array truncation.
    :param dataset: 2D np.array
    :param truncation_point: int of truncation index to test
    :return: np.array
    """
    assert (len(dataset.shape) == 2)  # Assert data_column is a 2D numpy array
    return dataset[-1 * truncation_point:]


def truncate_hdf5(hdf5_file, truncation_point):
    hdf5 = hdf5_file
    truncate_column(np.asarray(hdf5['admin/dates_index']), truncation_point)
    truncate_dataset(np.asarray(hdf5['data/clean']), truncation_point)
    hdf5.close()


if __name__ == '__main__':
    import unittest


    class UnitTester(unittest.TestCase):
        def setUp(self):
            pass

        def test_forward_fill(self):
            test_data_column_1 = np.array([np.nan, 2, 3, np.nan, np.nan], dtype=np.float32)
            solution_1 = np.array([np.nan, 2, 3, 3, 3], dtype=np.float32)
            test_result_1 = forward_fill(test_data_column_1)

            self.assertEqual(test_result_1.shape, (5,))
            self.assertEqual(test_result_1.dtype, np.float32)
            np.testing.assert_array_almost_equal(test_result_1, solution_1)

            test_data_column_2 = np.array([1, np.nan, np.nan, 4.5, np.nan], dtype=np.float32)
            solution_2 = np.array([1, 1, 1, 4.5, 4.5], dtype=np.float32)
            test_result_2 = forward_fill(test_data_column_2)
            self.assertEqual(test_result_2.shape, (5,))
            self.assertEqual(test_result_2.dtype, np.float32)
            np.testing.assert_array_almost_equal(test_result_2, solution_2)

        def test_time_scale(self):
            # monthly
            test_data_column = np.array([5, 10, 15, 5], dtype=np.float32)
            test_date_column = np.array(
                [dt.date(1967, 4, 1), dt.date(1967, 5, 1), dt.date(1967, 6, 2), dt.date(2017, 4, 1)])
            solution = np.empty(shape=(601,), dtype=np.float32)
            solution[:] = np.nan
            solution[0] = 5
            solution[1] = 10
            solution[2] = np.nan
            solution[-1] = 5
            test_result = time_scale(test_data_column, test_date_column)

            self.assertEqual(test_result.shape, (601,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal(test_result, solution)

            # daily
            test_data_column = np.array([10, 20, 50, 5], dtype=np.float32)
            test_date_column = np.array(
                [dt.date(1967, 4, 1), dt.date(1967, 4, 2), dt.date(1967, 6, 1), dt.date(2017, 4, 1)])
            solution = np.empty(shape=(601,), dtype=np.float32)
            solution[:] = np.nan
            solution[0] = 15
            solution[2] = 50
            solution[-1] = 5
            test_result = time_scale(test_data_column, test_date_column)

            self.assertEqual(test_result.shape, (601,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal(test_result, solution)

            # weekly
            test_data_column = np.array([10, 20, 30], dtype=np.float32)
            test_date_column = np.array([dt.date(1967, 4, 1), dt.date(1967, 4, 8), dt.date(1967, 4, 15)])
            solution = np.empty(shape=(601,), dtype=np.float32)
            solution[:] = np.nan
            solution[0] = float((10 + 20 + 30) / 3)
            test_result = time_scale(test_data_column, test_date_column)

            self.assertEqual(test_result.shape, (601,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal(test_result, solution)

            # quarterly
            test_data_column = np.array([30, 20, 10], dtype=np.float32)
            test_date_column = np.array([dt.date(1967, 4, 1), dt.date(1967, 7, 1), dt.date(1967, 10, 1)])
            solution = np.empty(shape=(601,), dtype=np.float32)
            solution[:] = np.nan
            solution[0] = 30
            solution[3] = 20
            solution[6] = 10
            test_result = time_scale(test_data_column, test_date_column)

            self.assertEqual(test_result.shape, (601,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal(test_result, solution)

            # annually
            test_data_column = np.array([30, 20, 10], dtype=np.float32)
            test_date_column = np.array([dt.date(1968, 1, 1), dt.date(1969, 1, 1), dt.date(1970, 1, 1)])
            solution = np.empty(shape=(601,), dtype=np.float32)
            solution[:] = np.nan
            solution[9] = 30
            solution[21] = 20
            solution[33] = 10
            test_result = time_scale(test_data_column, test_date_column)

            self.assertEqual(test_result.shape, (601,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal(test_result, solution)

        def test_trim_column(self):
            test_data_column = np.empty(shape=(700,), dtype=np.float32)
            solution = np.empty(shape=(601,), dtype=np.float32)
            test_result = trim_column(test_data_column)

            self.assertEqual(test_result.shape, (601,))
            self.assertEqual(test_result.dtype, np.float32)

        def test_truncate_loss(self):
            test_data_column = np.empty(shape=(601, 5), dtype=np.float32)
            test_data_column[:] = 1
            test_data_column[0, :] = np.nan
            test_data_column[1, :] = np.nan
            test_data_column[2, 2:3] = np.nan

            truncate_loss(test_data_column, isTest=True)

            with open('truncate_loss_UT.csv', 'r') as f:
                reader = csv.reader(f)
                percent_on_date = {}
                for row in reader:
                    date, percent = row[0].strip(), row[1].strip()
                    percent_on_date[date] = percent

            self.assertEqual(percent_on_date['1967-04-01'], '100.0%')
            self.assertEqual(percent_on_date['1967-05-01'], '100.0%')
            self.assertEqual(percent_on_date['1967-06-01'], '20.0%')

        def test_forward_fill_loss(self):
            test_data_column_raw = np.empty(shape=(601, 5), dtype=np.float32)
            test_data_column_raw[:] = 1
            test_data_column_raw[:, 0] = np.nan
            test_data_column_raw[0, 0] = 2
            test_data_column_raw[:, 1] = np.nan
            test_data_column_raw[0, 0] = 2

            test_data_column_clean = np.empty_like(test_data_column_raw)

            for i in range(0, 5, 1):
                test_data_column_clean[:, i] = forward_fill(test_data_column_raw[:, i])

            forward_fill_loss(test_data_column_raw, test_data_column_clean, isTest=True)

            with open('ffill_loss_month_UT.csv', 'r') as f:
                reader = csv.reader(f)
                percent_on_date = {}
                for row in reader:
                    date, percent = row[0].strip(), row[1].strip()
                    percent_on_date[date] = percent

            self.assertEqual(percent_on_date['1967-04-01'], '20.0%')
            self.assertEqual(percent_on_date['1967-05-01'], '40.0%')
            self.assertEqual(percent_on_date['1967-06-01'], '40.0%')

            with open('ffill_loss_feature_UT.csv', 'r') as f:
                reader = csv.reader(f)
                percent_on_feature = {}
                for row in reader:
                    feature, percent = row[0].strip(), row[1].strip()
                    percent_on_feature[feature] = percent

            self.assertEqual(percent_on_feature['1'], '99.83361064891847%')
            self.assertEqual(percent_on_feature['2'], '100.0%')
            self.assertEqual(percent_on_feature['3'], '0.0%')

        def test_truncate_column(self):
            test_data_column = np.empty(shape=(601,), dtype=np.float32)
            test_data_column[:] = 0
            test_data_column[-5:] = 1

            solution = np.empty(shape=(5,), dtype=np.float32)
            solution[:] = 1
            test_result = truncate_column(test_data_column, 5)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal(test_result, solution)

        def test_truncate_dataset(self):
            test_dataset = np.empty(shape=(601, 2), dtype=np.float32)
            test_dataset[:, 0] = 0
            test_dataset[-5:, 0] = 1
            test_dataset[:, 1] = 1
            test_dataset[-5:, 1] = 2

            solution = np.empty(shape=(5, 2), dtype=np.float32)
            solution[:, 0] = 1
            solution[:, 1] = 2
            test_result = truncate_dataset(test_dataset, 5)

            self.assertEqual(test_result.shape, (5, 2))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal(test_result, solution)

        def tearDown(self):
            pass


    unittest.main(verbosity=2)
