import datetime as dt

import numpy as np


def forward_fill(data_column):
    prev = np.arange(len(data_column))
    prev[data_column == 0] = 0
    prev = np.maximum.accumulate(prev)
    filled = data_column[prev]
    filled[filled == 0] = np.nan
    return filled


def time_scale(data_column, date_column, freq=None):
    # Removed redundancies, as they still provided accurate calculations even with different 'intervals'.

    if freq == 'monthly':
        scaled_data = np.empty(shape=(601,), dtype=np.float32)
        scaled_data[:] = 0

        start_date = np.datetime64('1967-04')
        date_list = []
        for i in range(0, 601, 1):
            date_list.append((start_date + np.timedelta64(i, 'M')).astype(dt.datetime))

        date_list = np.asarray(date_list)

        indices_dl = np.arange(date_list.shape[0])[np.in1d(date_list, date_column)]
        indices_dc = np.arange(date_column.shape[0])[np.in1d(date_column, date_list)]

        for i in range(0, len(indices_dl), 1):
            scaled_data[indices_dl[i]] = data_column[indices_dc[i]]

        return truncate_column(scaled_data)

    elif freq == 'daily':
        scaled_data = np.empty(shape=(18264,), dtype=np.float32)
        scaled_data[:] = 0

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

            scaled_data[i] = np.average(scaled_data[start:stop])

        return truncate_column(scaled_data)

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


def truncate_column(data_column, max_length=601):
    """
    Abstraction of numpy slicing operation for single column truncation.
    :param data_column: 1D np.array
    :param max_length: int of truncation index
    :return: np.array
    """
    assert (len(data_column.shape) == 1)  # Assert data_column is a 1D numpy array
    return data_column[:max_length]


def truncate_loss(dataset, truncation_point):
    """
    Returns percentage of columns with np.nan values; this indicates what percentage of features do not extend to the
    index specified.  (All other np.nan values should be forward filled)
    :param dataset: 2D np.array
    :param truncation_point: int of truncation index to test
    :return: float [0,1]
    """
    """
    Broken down example using code from
    http://stackoverflow.com/questions/12995937/count-all-values-in-a-matrix-greater-than-a-value
    consider_row = dataset[truncation_point]
    num_match = len(np.where(consider_row == np.nan)[0])
    num_total = len(consider_row)
    return num_match/num_total
    """
    # Adapted to one-line
    return len(np.where(dataset[truncation_point] == np.nan)[0]) / len(dataset[truncation_point])


def truncate_dataset(dataset, truncation_point):
    """
    Abstraction of numpy slicing operation for 2D array truncation.
    :param dataset: 2D np.array
    :param truncation_point: int of truncation index to test
    :return: np.array
    """
    assert (len(dataset.shape) == 2)  # Assert data_column is a 2D numpy array
    return dataset[:truncation_point]


if __name__ == '__main__':
    import unittest


    class UnitTester(unittest.TestCase):
        def setUp(self):
            pass

        def test_forward_fill(self):
            test_data_column_1 = np.array([0, 2, 3, 0, 0], dtype=np.float32)
            solution_1 = np.array([np.nan, 2, 3, 3, 3], dtype=np.float32)
            test_result_1 = forward_fill(test_data_column_1)

            self.assertEqual(test_result_1.shape, (5,))
            self.assertEqual(test_result_1.dtype, np.float32)
            np.testing.assert_array_equal(test_result_1, solution_1)

            test_data_column_2 = np.array([1, 0, 0, 4.5, 0], dtype=np.float32)
            solution_2 = np.array([1, 1, 1, 4.5, 4.5], dtype=np.float32)
            test_result_2 = forward_fill(test_data_column_2)

            self.assertEqual(test_result_2.shape, (5,))
            self.assertEqual(test_result_2.dtype, np.float32)
            np.testing.assert_array_equal(test_result_2, solution_2)

        def test_time_scale(self):
            # monthly
            test_data_column = np.array([5, 10, 15, 5], dtype=np.float32)
            test_date_column = np.array(
                [dt.date(1967, 4, 1), dt.date(1967, 5, 1), dt.date(1967, 6, 2), dt.date(2017, 4, 1)])
            solution = np.empty(shape=(601,), dtype=np.float32)
            solution[:] = 0
            solution[0] = 5
            solution[1] = 10
            solution[2] = 0
            solution[-1] = 5
            test_result = time_scale(test_data_column, test_date_column)

            self.assertEqual(test_result.shape, (601,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

            # daily
            test_data_column = np.array([10, 20, 50, 5], dtype=np.float32)
            test_date_column = np.array(
                [dt.date(1967, 4, 1), dt.date(1967, 4, 2), dt.date(1967, 6, 1), dt.date(2017, 4, 1)])
            solution = np.empty(shape=(601,), dtype=np.float32)
            solution[:] = 0
            solution[0] = 1
            solution[2] = float(5 / 3)
            solution[-1] = float(5)
            test_result = time_scale(test_data_column, test_date_column)

            self.assertEqual(test_result.shape, (601,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

            # weekly
            test_data_column = np.array([10, 20, 30], dtype=np.float32)
            test_date_column = np.array([dt.date(1967, 4, 1), dt.date(1967, 4, 8), dt.date(1967, 4, 15)])
            solution = np.empty(shape=(601,), dtype=np.float32)
            solution[:] = 0
            solution[0] = float((10 + 20 + 30) / 30)
            test_result = time_scale(test_data_column, test_date_column)

            self.assertEqual(test_result.shape, (601,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

            # quarterly
            test_data_column = np.array([30, 20, 10], dtype=np.float32)
            test_date_column = np.array([dt.date(1967, 4, 1), dt.date(1967, 7, 1), dt.date(1967, 10, 1)])
            solution = np.empty(shape=(601,), dtype=np.float32)
            solution[:] = 0
            solution[0] = 30
            solution[3] = 20
            solution[6] = 10
            test_result = time_scale(test_data_column, test_date_column)

            self.assertEqual(test_result.shape, (601,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

            # annually
            test_data_column = np.array([30, 20, 10], dtype=np.float32)
            test_date_column = np.array([dt.date(1968, 1, 1), dt.date(1969, 1, 1), dt.date(1970, 1, 1)])
            solution = np.empty(shape=(601,), dtype=np.float32)
            solution[:] = 0
            solution[9] = 30
            solution[21] = 20
            solution[33] = 10
            test_result = time_scale(test_data_column, test_date_column)

            self.assertEqual(test_result.shape, (601,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

        def test_truncate_column(self):
            test_data_column = np.empty(shape=(700,), dtype=np.float32)
            solution = np.empty(shape=(601,), dtype=np.float32)
            test_result = truncate_column(test_data_column)

            self.assertEqual(test_result.shape, (601,))
            self.assertEqual(test_result.dtype, np.float32)

        def test_truncate_loss(self):
            # ISSUE: Not certain if I tested this one correctly.
            test_data_column = np.empty(shape=(300, 2), dtype=np.float32)
            test_data_column[:150] = np.nan
            solution = np.empty(shape=(300, 2), dtype=np.float32)
            solution[:150] = np.nan
            test_result = truncate_loss(test_data_column, 150)

            self.assertEqual(test_result, float(0))

        def test_truncate_dataset(self):
            test_data_column = np.empty(shape=(700, 2), dtype=np.float32)
            solution = np.empty(shape=(601, 2), dtype=np.float32)
            test_result = truncate_dataset(test_data_column, 601)

            self.assertEqual(test_result.shape, (601, 2))
            self.assertEqual(test_result.dtype, np.float32)

        def tearDown(self):
            pass


    unittest.main(verbosity=2)
