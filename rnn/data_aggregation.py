import h5py
import numpy as np


def aggregate_rnn_data():
    # TODO open FREDcast.hdf5 and a new file, rnn_data.hdf5
    # Copy the admin group over, without sample_values_index
    # Then expand values_index for the aggregation to follow...

    # TODO Aggregate all 12 normalization combinations into one dataset
    # (could just hstack, but that might not fit in memory)
    # for example, if the initial datasets were (601, 100000)
    # the aggregate should be (601, initial.shape[1] * 12) or (601, 1200000)
    # hint: probably a good assertion to ensure correctness there

    # TODO Split this into train/test just like in data_interfaces

    # TODO Save it all to hdf5

    raise NotImplementedError