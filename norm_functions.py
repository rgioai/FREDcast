import numpy as np
import h5py

def zero_one(data_column):
    # ((x - x_min)/(x_max - x_min))

    copy = np.empty_like(data_column)
    copy[:] = data_column

    max = np.nanmax(copy)
    min = np.nanmin(copy)

    def norm(n):
        if not np.isnan(n):
            return (n - min) / (max - min)
        return n

    vfunc = np.vectorize(norm)
    return vfunc(copy)


def percent_change(data_column):
    # (x[i]): ((x[i] - x[i-1])/x[i-1]); if i == 0: x[i] = 0

    copy = np.empty_like(data_column)
    copy[:] = data_column

    n = np.empty_like(data_column)
    n[:] = data_column

    try:
        start_point = np.argwhere(np.isnan(copy))[-1][0] + 1
    except IndexError:
        start_point = 0

    for i in range(start_point, len(copy)):
        if i == start_point:
            copy[i] = 0
        else:
            if n[i - 1] == 0:
                copy[i] = 0
            else:
                copy[i] = ((n[i] - n[i - 1]) / n[i - 1])
    return copy


def normal_dist(data_column):
    # (x - x_mean)/x_std

    copy = np.empty_like(data_column)
    copy[:] = data_column

    mean = np.nanmean(copy)
    std = np.nanstd(copy)

    def norm(n):
        if not np.isnan(n):
            return (n - mean) / std
        return n

    vfunc = np.vectorize(norm)
    return vfunc(copy)


def linear_residual(data_column, second_norm=None):
    # line from first to last for column, generate inference_function
    # x - inference_function(x)

    copy = np.empty_like(data_column)
    copy[:] = data_column

    y = copy[~np.isnan(copy)]
    x = np.arange(1, len(y) + 1).astype(np.float32)

    line = np.polyfit(x, y, 1)

    def norm(n):
        if not np.isnan(n):
            return n - ((line[0] * n) + line[1])
        return n

    vfunc = np.vectorize(norm)
    first_norm = vfunc(copy)

    if second_norm is not None:
        return second_norm(first_norm).astype(np.float32)
    else:
        return first_norm.astype(np.float32)


def exp_residual(data_column, second_norm=None):
    # CAGR for column, generate inference_function
    # x - inference_function(x)

    copy = np.empty_like(data_column)
    copy[:] = data_column

    year_dif = data_column.size / 12

    def cagr_func(f, l, n):
        return ((l / f) ** (1 / n)) - 1

    try:
        start_point = np.argwhere(np.isnan(copy))[-1][0] + 1
    except IndexError:
        start_point = 0

    first = copy[start_point]
    last = copy[-1]
    cagr = cagr_func(first, last, year_dif)

    def norm(n):
        if not np.isnan(n):
            return n - cagr
        return n

    vfunc = np.vectorize(norm)
    first_norm = vfunc(copy)

    if second_norm is not None:
        return second_norm(first_norm).astype(np.float32)
    else:
        return first_norm.astype(np.float32)


def gdp_residual(data_column, second_norm=None):
    # (x[i]): ((x[i] - x[i-1])/x[i-1]); if i == 0: x[i] = 0
    # x[i] - ((gdp[i] - gdp[i-1])/gdp[i-1]); if i == 0: gdp[i] = 0

    copy = np.empty_like(data_column)
    copy[:] = data_column

    n = percent_change(copy)

    # hdf5 file with gdp data
    hdf5 = h5py.File('FREDcast.hdf5')
    gdp = np.asarray(hdf5['admin/gdp'])
    hdf5.close()

    try:
        start_point = np.argwhere(np.isnan(copy))[-1][0] + 1
    except IndexError:
        start_point = 0

    for i in range(start_point, len(copy)):
        if i == start_point:
            copy[i] = 0
        else:
            if gdp[i - 1] == 0:
                copy[i] = 0
            else:
                copy[i] = (n[i] - ((gdp[i] - gdp[i - 1]) / gdp[i - 1]))

    if second_norm is not None:
        return second_norm(copy).astype(np.float32)
    else:
        return copy.astype(np.float32)


def normalize_dataset(dataset, norm_fn, second_norm=None):
    """
    In pseudocode:
    If norm_fn is one of the residuals, use lambda to create a new function that takes only data_column as input.
    Then use np.apply_along_axis to map norm_fn across the vertical axis of the dataset.
    Return the new array
    """
    copy = np.empty_like(dataset)
    copy[:] = dataset

    if norm_fn is not None:
        for i in range(0, copy.shape[1]):
            if not np.count_nonzero(np.isnan(copy[:, i])) == copy[:, i].size:
                if second_norm is None:
                    copy[:, i] = norm_fn(copy[:, i])
                else:
                    copy[:, i] = norm_fn(copy[:, i], second_norm)
        return copy
    else:
        raise ValueError('Missing dataset value!')


if __name__ == '__main__':
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
            test_data_column_a = np.array([1, 5, 7, 8, 11], dtype=np.float32)
            solution = np.array([-0.8, -6, -8.6, -9.9, -13.8], dtype=np.float32)
            test_result = linear_residual(test_data_column_a, second_norm=None)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

            test_data_column_b = np.array([1, 5, 7, 8, 11], dtype=np.float32)
            solution = np.array([1, 0.6, 0.4, 0.3, 0], dtype=np.float32)
            test_result = linear_residual(test_data_column_b, second_norm=zero_one)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

            test_data_column_c = np.array([1, 5, 7, 8, 11], dtype=np.float32)
            solution = np.array([0, 6.5, 0.433333, 0.151163, 0.393939], dtype=np.float32)
            test_result = linear_residual(test_data_column_c, second_norm=percent_change)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=14)

            test_data_column_d = np.array([1, 5, 7, 8, 11], dtype=np.float32)
            solution = np.array([1.625209, 0.42135, -0.180579, -0.481543, -1.38447], dtype=np.float32)
            test_result = linear_residual(test_data_column_d, second_norm=normal_dist)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=274)

        def test_exp_residual(self):
            test_data_column_a = np.array([1, 5, 7, 8, 11], dtype=np.float32)
            solution = np.array([-313.749329, -309.749329, -307.749329, -306.749329, -303.749329], dtype=np.float32)
            test_result = exp_residual(test_data_column_a, second_norm=None)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

            test_data_column_b = np.array([1, 5, 7, 8, 11], dtype=np.float32)
            solution = np.array([0, 0.4, 0.6, 0.7, 1], dtype=np.float32)
            test_result = exp_residual(test_data_column_b, second_norm=zero_one)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

            test_data_column_c = np.array([1, 5, 7, 8, 11], dtype=np.float32)
            solution = np.array([0, -0.012749, -0.006457, -0.003249, -0.00978], dtype=np.float32)
            test_result = exp_residual(test_data_column_c, second_norm=percent_change)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=1708)

            test_data_column_d = np.array([1, 5, 7, 8, 11], dtype=np.float32)
            solution = np.array([-1.625209, -0.42135, 0.180579, 0.481543, 1.384437], dtype=np.float32)
            test_result = exp_residual(test_data_column_d, second_norm=normal_dist)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=16)

        def test_gdp_residual(self):
            test_data_column_a = np.array([1000, 2000, 4000, 3500, 4500], dtype=np.float32)
            solution = np.array([0, 1, 1, -0.143212, 0.285714], dtype=np.float32)
            test_result = gdp_residual(test_data_column_a, second_norm=None)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=19)

            test_data_column_b = np.array([1000, 2000, 4000, 3500, 4500], dtype=np.float32)
            solution = np.array([0.125271, 1, 1, 0, 0.375194], dtype=np.float32)
            test_result = gdp_residual(test_data_column_b, second_norm=zero_one)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=26)

            test_data_column_c = np.array([1000, 2000, 4000, 3500, 4500], dtype=np.float32)
            solution = np.array([0, 0, 0, -1.143212, -2.995048], dtype=np.float32)
            test_result = gdp_residual(test_data_column_c, second_norm=percent_change)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=2)

            test_data_column_d = np.array([1000, 2000, 4000, 3500, 4500], dtype=np.float32)
            solution = np.array([-0.880534, 1.174385, 1.174385, -1.174822, -0.293414], dtype=np.float32)
            test_result = gdp_residual(test_data_column_d, second_norm=normal_dist)

            self.assertEqual(test_result.shape, (5,))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=3)

        def test_normalize_dataset(self):
            test_data_column_1 = np.array([10, 20, 30], dtype=np.float32)
            test_data_column_1 = test_data_column_1.reshape(test_data_column_1.size, 1)
            test_data_column_2 = np.array([20, 30, 40], dtype=np.float32)
            test_data_column_2 = test_data_column_2.reshape(test_data_column_2.size, 1)
            test_dataset_a = np.hstack((test_data_column_1, test_data_column_2))

            # linear residual
            solution_1 = np.array([-90, -180, -270], dtype=np.float32)
            solution_1 = solution_1.reshape(solution_1.size, 1)
            solution_2 = np.array([-190, -280, -370], dtype=np.float32)
            solution_2 = solution_2.reshape(solution_2.size, 1)
            solution = np.hstack((solution_1, solution_2))

            self.assertRaises(ValueError, lambda: normalize_dataset(test_dataset_a, None))
            test_result = normalize_dataset(test_dataset_a, linear_residual)

            self.assertEqual(test_result.shape, (3, 2))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

            # linear residual + zero_one
            test_dataset_b = np.hstack((test_data_column_1, test_data_column_2))

            solution_1 = np.array([1, 0.5, 0], dtype=np.float32)
            solution_1 = solution_1.reshape(solution_1.size, 1)
            solution_2 = np.array([1, 0.5, 0], dtype=np.float32)
            solution_2 = solution_2.reshape(solution_2.size, 1)
            solution = np.hstack((solution_1, solution_2))

            test_result = normalize_dataset(test_dataset_b, linear_residual, zero_one)

            self.assertEqual(test_result.shape, (3, 2))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

            # linear residual + percent_change
            test_dataset_c = np.hstack((test_data_column_1, test_data_column_2))

            solution_1 = np.array([0, 1, 0.5], dtype=np.float32)
            solution_1 = solution_1.reshape(solution_1.size, 1)
            solution_2 = np.array([0, 0.473684, 0.321429], dtype=np.float32)
            solution_2 = solution_2.reshape(solution_2.size, 1)
            solution = np.hstack((solution_1, solution_2))

            test_result = normalize_dataset(test_dataset_c, linear_residual, percent_change)

            self.assertEqual(test_result.shape, (3, 2))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=15)

            # linear residual + normal_dist
            test_dataset_d = np.hstack((test_data_column_1, test_data_column_2))

            solution_1 = np.array([1.224745, -3.867705e-16, -1.224745], dtype=np.float32)
            solution_1 = solution_1.reshape(solution_1.size, 1)
            solution_2 = np.array([1.224745, 0, -1.224745], dtype=np.float32)
            solution_2 = solution_2.reshape(solution_2.size, 1)
            solution = np.hstack((solution_1, solution_2))

            test_result = normalize_dataset(test_dataset_d, linear_residual, normal_dist)

            self.assertEqual(test_result.shape, (3, 2))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=1)

            # exp residual
            test_dataset_e = np.hstack((test_data_column_1, test_data_column_2))

            solution_1 = np.array([-70, -60, -50], dtype=np.float32)
            solution_1 = solution_1.reshape(solution_1.size, 1)
            solution_2 = np.array([5, 15, 25], dtype=np.float32)
            solution_2 = solution_2.reshape(solution_2.size, 1)
            solution = np.hstack((solution_1, solution_2))

            self.assertRaises(ValueError, lambda: normalize_dataset(test_dataset_a, None))
            test_result = normalize_dataset(test_dataset_e, exp_residual)

            self.assertEqual(test_result.shape, (3, 2))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

            # exp residual + zero_one
            test_dataset_f = np.hstack((test_data_column_1, test_data_column_2))

            solution_1 = np.array([0, 0.5, 1], dtype=np.float32)
            solution_1 = solution_1.reshape(solution_1.size, 1)
            solution_2 = np.array([0, 0.5, 1], dtype=np.float32)
            solution_2 = solution_2.reshape(solution_2.size, 1)
            solution = np.hstack((solution_1, solution_2))

            test_result = normalize_dataset(test_dataset_f, exp_residual, zero_one)

            self.assertEqual(test_result.shape, (3, 2))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_equal(test_result, solution)

            # exp residual + percent_change
            test_dataset_g = np.hstack((test_data_column_1, test_data_column_2))

            solution_1 = np.array([0, -0.142857, -0.166667], dtype=np.float32)
            solution_1 = solution_1.reshape(solution_1.size, 1)
            solution_2 = np.array([0, 2, 0.666667], dtype=np.float32)
            solution_2 = solution_2.reshape(solution_2.size, 1)
            solution = np.hstack((solution_1, solution_2))

            test_result = normalize_dataset(test_dataset_g, exp_residual, percent_change)

            self.assertEqual(test_result.shape, (3, 2))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=22)

            # exp residual + normal_dist
            test_dataset_h = np.hstack((test_data_column_1, test_data_column_2))

            solution_1 = np.array([-1.224745, 0, 1.224745], dtype=np.float32)
            solution_1 = solution_1.reshape(solution_1.size, 1)
            solution_2 = np.array([-1.224745, 0, 1.224745], dtype=np.float32)
            solution_2 = solution_2.reshape(solution_2.size, 1)
            solution = np.hstack((solution_1, solution_2))

            test_result = normalize_dataset(test_dataset_h, exp_residual, normal_dist)

            self.assertEqual(test_result.shape, (3, 2))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=1)

            # gdp residual
            test_dataset_i = np.hstack((test_data_column_1, test_data_column_2))

            solution_1 = np.array([0, 1, 0.5], dtype=np.float32)
            solution_1 = solution_1.reshape(solution_1.size, 1)
            solution_2 = np.array([0, 0.5, 0.333333], dtype=np.float32)
            solution_2 = solution_2.reshape(solution_2.size, 1)
            solution = np.hstack((solution_1, solution_2))

            self.assertRaises(ValueError, lambda: normalize_dataset(test_dataset_a, None))
            test_result = normalize_dataset(test_dataset_i, gdp_residual)

            self.assertEqual(test_result.shape, (3, 2))
            self.assertEqual(test_result.dtype, np.float32)

            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=12)

            # gdp residual + zero_one
            test_dataset_j = np.hstack((test_data_column_1, test_data_column_2))

            solution_1 = np.array([0, 1, 0.5], dtype=np.float32)
            solution_1 = solution_1.reshape(solution_1.size, 1)
            solution_2 = np.array([0, 1, 0.666667], dtype=np.float32)
            solution_2 = solution_2.reshape(solution_2.size, 1)
            solution = np.hstack((solution_1, solution_2))

            test_result = normalize_dataset(test_dataset_j, gdp_residual, zero_one)

            self.assertEqual(test_result.shape, (3, 2))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=5)

            # gdp residual + percent_change
            test_dataset_k = np.hstack((test_data_column_1, test_data_column_2))

            solution_1 = np.array([0, 0, -0.5], dtype=np.float32)
            solution_1 = solution_1.reshape(solution_1.size, 1)
            solution_2 = np.array([0, 0, -0.333333], dtype=np.float32)
            solution_2 = solution_2.reshape(solution_2.size, 1)
            solution = np.hstack((solution_1, solution_2))

            test_result = normalize_dataset(test_dataset_k, gdp_residual, percent_change)

            self.assertEqual(test_result.shape, (3, 2))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=11)

            # gdp residual + normal_dist
            test_dataset_l = np.hstack((test_data_column_1, test_data_column_2))

            solution_1 = np.array([-1.224745, 1.224745, 0], dtype=np.float32)
            solution_1 = solution_1.reshape(solution_1.size, 1)
            solution_2 = np.array([-1.336306, 1.069045, 0.267261], dtype=np.float32)
            solution_2 = solution_2.reshape(solution_2.size, 1)
            solution = np.hstack((solution_1, solution_2))

            test_result = normalize_dataset(test_dataset_l, gdp_residual, normal_dist)

            self.assertEqual(test_result.shape, (3, 2))
            self.assertEqual(test_result.dtype, np.float32)
            np.testing.assert_array_almost_equal_nulp(test_result, solution, nulp=8)

        def tearDown(self):
            pass


    unittest.main(verbosity=2)
