import numpy as np


# TODO Review the block comments. As long as they make sense, delete them.


def forward_fill(data_column):
    # By filling our data on top of an array of np.nan values, we already have the right nan setup.
    # Now, for every nan, check if there is a previous real value.
    # If there is, use the most recent previous value in this cell.
    # TODO Implement
    # Check some examples at:
    # http://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    pass


def time_scale(data_column, freq=None):
    if freq == 'monthly':
        return truncate_column(data_column)
        # I've left the column truncation inside the time_scale function
        # rather than leave it at a higher scope.
    elif freq == 'daily':
        # TODO use average to smooth to monthly values
        pass
    elif freq == 'weekly':
        # TODO use average to smooth to monthly values
        pass
    elif freq == 'quarterly':
        # TODO each month of each quarter takes that quarter's values
        pass
    elif freq == 'annual':
        # TODO each month of each year takes that year's values
        pass
    elif freq is None:
        # TODO Determine frequency
        auto_freq = ''
        return time_scale(data_column, auto_freq)
    else:
        raise ValueError('Frequency value not recognized')
    """
    There's a lot of leeway here on how to actually implement these, but the best plan is probably allocating a new
    array of the desired size and iterating through the original, selectively copying the desired values over.  Any
    array resizing is going to do a reallocation at the C layer anyway, so might as well at the python level.
    Remember, a new array is a one liner with: np.empty(shape_tuple).fill(np.nan)
    """
    """
    Little note here about if/elif/else statements--can't remember if I've said anything before.  Go ahead and delete
    this block if I have/once it makes sense.  TL;DR is only use if - else when they accurately describe your problem.
    For example, (if condition is true) - (else a condition is not true).  We cover the entire space of possibility.
    Either something is true or it is not.  Using else as a catch-all for a mutable variable is a whole different
    issue.  In the code above, I could have used else instead of 'elif freq is None' because that's the last condition
    I expect.  But it is NOT the only possible condition.  freq could be any value (that's the nasty part of python)
    so we don't want to operate with it if freq isn't a value that we expect.

    By giving if/elif conditions for all expected conditions and using else as a catch-all error condition, we
    add fairly complicated type checking without much additional effort.
    """


def truncate_column(data_column, max_length=600):
    """
    Abstraction of numpy slicing operation for single column truncation.
    :param data_column: 1D np.array
    :param max_length: int of truncation index
    :return: np.array
    """
    assert(len(data_column.shape) == 1)  # Assert data_column is a 1D numpy array
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
    assert(len(data_column.shape) == 2)  # Assert data_column is a 2D numpy array
    return dataset[:truncation_point]


if __name__ == '__main__':
    # TODO Unit testing
    pass
