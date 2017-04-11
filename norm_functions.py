import h5py
import numpy as np


def zero_one(data_column):
    # ((x - x_min)/(x_max - x_min))

    max = np.amax(data_column)
    min = np.amin(data_column)

    def norm(n):
        return (n - min) / (max - min)

    vfunc = np.vectorize(norm)
    return vfunc(data_column)


def percent_change(data_column):
    # (x[i]): ((x[i] - x[i-1])/x[i-1]); if i == 0: x[i] = 0

    n = np.empty_like(data_column)
    n[:] = data_column
    for i in range(0, len(data_column)):
        if i == 0:
            data_column[i] = 0
        else:
            data_column[i] = ((n[i] - n[i - 1]) / n[i - 1])
    return data_column


def normal_dist(data_column):
    # (x - x_mean)/x_std

    mean = np.mean(data_column)
    std = np.std(data_column)

    def norm(n):
        return (n - mean) / std

    vfunc = np.vectorize(norm)
    return vfunc(data_column)


def linear_residual(data_column, second_norm=None):
    # line from first to last for column, generate inference_function
    # x - inference_function(x)

    x = np.arange(1, len(data_column) + 1).astype(np.float32)
    y = data_column

    line = np.polyfit(x, y, 1)

    def norm(n):
        return n - ((line[0] * n) + line[1])

    vfunc = np.vectorize(norm)
    first_norm = vfunc(data_column)

    if second_norm is not None:
        return second_norm(first_norm).astype(np.float32)
    else:
        return first_norm.astype(np.float32)


def exp_residual(data_column, date_column, second_norm=None):
    # CAGR for column, generate inference_function
    # x - inference_function(x)

    year_dif = float(str(date_column[-1])[2:6]) - int(str(date_column[0])[2:6])

    def cagr_func(f, l, n):
        return ((l / f) ** (1 / n)) - 1

    cagr = cagr_func(data_column[0], data_column[-1], year_dif)

    def norm(n):
        return n - cagr

    vfunc = np.vectorize(norm)
    first_norm = vfunc(data_column)

    if second_norm is not None:
        return second_norm(first_norm).astype(np.float32)
    else:
        return first_norm.astype(np.float32)


def gdp_residual(data_column, second_norm=None):
    # (x[i]): ((x[i] - x[i-1])/x[i-1]); if i == 0: x[i] = 0
    # x[i] - ((gdp[i] - gdp[i-1])/gdp[i-1]); if i == 0: gdp[i] = 0

    n = percent_change(data_column)

    # hdf5 file with gdp data
    f = h5py.File('y_values.h5', 'r')
    gdp = np.asarray(f['gdp'])

    for i in range(0, len(data_column)):
        if i == 0:
            data_column[i] = 0
        else:
            data_column[i] = (n[i] - ((gdp[i] - gdp[i - 1]) / gdp[i - 1]))

    if second_norm is not None:
        return second_norm(data_column).astype(np.float32)
    else:
        return data_column.astype(np.float32)


def normalize_dataset(dataset, norm_fn, second_norm=None):
    """
    In pseudocode:
    If norm_fn is one of the residuals, use lambda to create a new function that takes only data_column as input.
    Then use np.apply_along_axis to map norm_fn across the vertical axis of the dataset.
    Return the new array
    """
    if norm_fn is not None:
        if norm_fn is 'linear_residual':
            return np.apply_along_axis(lambda x: linear_residual(x, second_norm), dataset)
        elif norm_fn is 'exp_residual':
            return np.apply_along_axis(lambda x: exp_residual(x, second_norm), dataset)
        elif norm_fn is 'gdp_residual':
            return np.apply_along_axis(lambda x: gdp_residual(x, second_norm), dataset)
    else:
        raise ValueError('Missing dataset value!')


if __name__ == '__main__':
    # TODO Unit testing for all functions
    import unittest


    class UnitTester(unittest.TestCase):
        def setUp(self):
            pass

        def test_zero_one(self):
            test_data_column = np.array([1, 2, 3, 4, 5], dtype=np.float32)
            solution = np.array([0, 0.25, 0.5, 0.75, 1], dtype=np.float32)
            test_result = zero_one(test_data_column)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

        def test_percent_change(self):
            test_data_column = np.array([1, 2, 3, 4, 5], dtype=np.float32)
            solution = np.array([0, 1, 0.5, 0.33333334, 0.25], dtype=np.float32)
            test_result = percent_change(test_data_column)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

        def test_normal_dist(self):
            test_data_column = np.array([1, 2, 3, 4, 5], dtype=np.float32)
            solution = np.array([-1.414214, -0.707107, 0, 0.707107, 1.414214], dtype=np.float32)
            test_result = normal_dist(test_data_column)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=4)

        def test_linear_residual(self):
            test_data_column = np.array([1, 5, 7, 8, 11], dtype=np.float32)
            solution = np.array([-0.8, -6, -8.6, -9.9, -13.8], dtype=np.float32)
            test_result = linear_residual(test_data_column, second_norm=None)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

            solution = np.array([1, 0.6, 0.4, 0.3, 0], dtype=np.float32)
            test_result = linear_residual(test_data_column, second_norm=zero_one)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

            solution = np.array([0, 6.5, 0.433333, 0.151163, 0.393939], dtype=np.float32)
            test_result = linear_residual(test_data_column, second_norm=percent_change)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=14)

            solution = np.array([1.625209, 0.42135, -0.180579, -0.481543, -1.38447], dtype=np.float32)
            test_result = linear_residual(test_data_column, second_norm=normal_dist)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=274)

        def test_exp_residual(self):
            test_data_column = np.array([1, 5, 7, 8, 11], dtype=np.float32)
            test_date_column = np.array(
                ['1967-04-30', '1967-07-31', '1967-10-31', '1968-01-31', '1969-04-30'], dtype='S10')

            #dates_file = h5py.File('y_dates.h5', 'r')
            #dates = dates_file['gdp']
            #print(np.asarray(dates))

            solution = np.array([-1.316625, 2.683375, 4.683375, 5.683375, 8.683375], dtype=np.float32)
            test_result = exp_residual(test_data_column, test_date_column, second_norm=None)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=2)

            solution = np.array([0, 0.4, 0.6, 0.7, 1], dtype=np.float32)
            test_result = exp_residual(test_data_column, test_date_column, second_norm=zero_one)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

            solution = np.array([0, -3.038071, 0.74533, 0.213521, 0.527855], dtype=np.float32)
            test_result = exp_residual(test_data_column, test_date_column, second_norm=percent_change)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=15)

            solution = np.array([-1.625209, -0.42135, 0.180579, 0.481543, 1.38447], dtype=np.float32)
            test_result = exp_residual(test_data_column, test_date_column, second_norm=normal_dist)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=274)

        def test_gdp_residual(self):
            test_data_column = np.array([1000, 2000, 4000, 3500, 4500], dtype=np.float32)
            solution = np.array([-849450, -1698450, -3396450, -2971950, -3820950], dtype=np.float32)
            test_result = linear_residual(test_data_column, second_norm=None)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

            solution = np.array([1, 0.714286, 0.142857, 0.285714, 0], dtype=np.float32)
            test_result = linear_residual(test_data_column, second_norm=zero_one)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=10)

            solution = np.array([0, 0.99947, 0.999735, -0.124983, 0.285671], dtype=np.float32)
            test_result = linear_residual(test_data_column, second_norm=percent_change)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=59)

            solution = np.array([1.53393, 0.766965, -0.766965, -0.383482, -1.150447], dtype=np.float32)
            test_result = linear_residual(test_data_column, second_norm=normal_dist)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=16)

        def test_normalize_dataset(self):
            pass

        def tearDown(self):
            pass


    unittest.main(verbosity=2)
