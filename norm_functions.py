import numpy as np


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
    pass


def percent_change(data_column):
    # TODO map to every value (x[i]): ((x[i] - x[i-1])/x[i-1]); if i == 0: x[i] = 0
    # TODO return the new array
    pass


def normal_dist(data_column):
    # TODO map to every value: (x - x_mean)/x_std
    # TODO return the new array
    pass


def linear_residual(data_column, second_norm=None):
    # TODO Calculate line from first to last for column, generate inference_function
    # TODO Map to every value: x - inference_function(x)
    # TODO Map second_norm to every value
    # TODO return the new array
    pass


def exp_residual(data_column, second_norm=None):
    # TODO Calculate CAGR for column, generate inference_function
    # TODO Map to every value: x - inference_function(x)
    # TODO Map second_norm to every value
    # TODO return the new array
    pass


def gdp_residual(data_column, second_norm=None):
    # TODO map to every value (x[i]): ((x[i] - x[i-1])/x[i-1]); if i == 0: x[i] = 0
    # TODO map to every value x[i] - ((gdp[i] - gdp[i-1])/gdp[i-1]); if i == 0: gdp[i] = 0
    # TODO map second_norm to every value
    # TODO return the new array
    pass


def normalize_dataset(dataset, norm_fn, second_norm=None):
    """
    In pseudocode:
    If norm_fn is one of the residuals, use lambda to create a new function that takes only data_column as input.
    Then use np.apply_along_axis to map norm_fn across the vertical axis of the dataset.
    Return the new array
    """
    pass


if __name__ == '__main__':
    import unittest
    # TODO Unit testing for all functions
    pass
