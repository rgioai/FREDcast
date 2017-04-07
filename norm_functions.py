import numpy as np
import h5py

def zero_one(data_column):
    # TODO map to every value: ((x - x_min)/(x_max - x_min))
    """
    In pseudocode:
    max = data_column.max
    min = data_column.min
    def norm(x):
        return (x - min)/(max - min)
    vfunc = np.vectorize(norm)
    return vfunc(data_column)
    """
    # TODO return the new array
    max = np.amax(data_column)
    min = np.amin(data_column)

    def norm(n):
        return (n - min) / (max - min)

    vfunc = np.vectorize(norm)
    return vfunc(data_column)


def percent_change(data_column):
    # TODO map to every value (x[i]): ((x[i] - x[i-1])/x[i-1]); if i == 0: x[i] = 0
    # TODO return the new array

    n = data_column
    for i in range(0, len(data_column)):
        if i == 0:
            data_column[i] = 0
        else:
            data_column[i] = ((n[i] - n[i - 1]) / n[i - 1])
    return data_column


def normal_dist(data_column):
    # TODO map to every value: (x - x_mean)/x_std
    # TODO return the new array
    mean = np.mean(data_column)
    std = np.std(data_column)

    def norm(n):
        return (n - mean) / std

    vfunc = np.vectorize(norm)
    return vfunc(data_column)


def linear_residual(data_column, second_norm=None):
    # TODO Calculate line from first to last for column, generate inference_function
    # TODO Map to every value: x - inference_function(x)
    # TODO Map second_norm to every value
    # TODO return the new array
    x = np.arange(1, len(data_column))
    y = data_column

    line = np.polyfit(x, y, 1)

    def norm(n):
        return n - ((line[0] * n) + line[1])

    vfunc = np.vectorize(norm)
    first_norm = vfunc(data_column)

    if second_norm is not None:
        return second_norm(first_norm)
    else:
        return first_norm


def exp_residual(data_column, date_column, second_norm=None):
    # TODO Calculate CAGR for column, generate inference_function
    # TODO Map to every value: x - inference_function(x)
    # TODO Map second_norm to every value
    # TODO return the new array

    year_dif = int(str(date_column[-1])[2:6]) - int(str(date_column[0])[2:6])

    def cagr_func(f, l, n):
        return ((l / f) ^ (1 / n)) - 1

    cagr = cagr_func(data_column[0], data_column[-1], year_dif)

    def norm(n):
        return n - cagr

    vfunc = np.vectorize(norm)
    first_norm = vfunc(data_column)

    if second_norm is not None:
        return second_norm(first_norm)
    else:
        return first_norm


def gdp_residual(data_column, second_norm=None):
    # TODO map to every value (x[i]): ((x[i] - x[i-1])/x[i-1]); if i == 0: x[i] = 0
    # TODO map to every value x[i] - ((gdp[i] - gdp[i-1])/gdp[i-1]); if i == 0: gdp[i] = 0
    # TODO map second_norm to every value
    # TODO return the new array
    n = percent_change(data_column)

    #hdf5 file with gdp data
    f = h5py.File('y_values.h5', 'r')
    gdp = np.asarray(f['gdp'])

    for i in range(0, len(data_column)):
        if i == 0:
            data_column[i] = 0
        else:
            data_column[i] = ((n[i] - n[i - 1]) / n[i - 1])
    return data_column


def normalize_dataset(dataset, norm_fn, second_norm=None):
    """
    In pseudocode:
    If norm_fn is one of the residuals, use lambda to create a new function that takes only data_column as input.
    Then use np.apply_along_axis to map norm_fn across the vertical axis of the dataset.
    Return the new array
    """
    pass


if __name__ == '__main__':
    # TODO Unit testing for all functions
    pass
