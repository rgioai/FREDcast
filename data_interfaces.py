import h5py
import numpy as np


def split_data():
    # TODO open FREDcast.hdf5 and a new file, split_data.hdf5
    # copy the admin group directly
    # then split each of the normalized datasets,
    # with all but the last 3 months going into training
    # and the last three months going into testing.
    # Be sure to check datatype and dset shape before returning.
    raise NotImplementedError


class Interface(object):
    def __init__(self, norm_fn, residual_fn, sample=False):
        # TODO If split_data.hdf5 does not exist, prompt the user
        # if yes, create it; otherwise quit
        self.norm_fn = norm_fn
        self.residual_fn = residual_fn
        self.sample = sample
        raise NotImplementedError

    def train_x(self):
        raise NotImplementedError

    def train_y(self):
        raise NotImplementedError

    def test_x(self):
        raise NotImplementedError

    def test_y(self):
        raise NotImplementedError

    def get_data(self):
        return self.train_x(), self.train_y(), self.test_x(), self.test_y()


class SKLearn_Interface(Interface):
    # TODO Verify that the init method of the parent class will run without an explicit call
    def train_x(self):
        raise NotImplementedError
        # TODO Return all train x in a numpy array from {sample}_{self.norm_fn}_{self.residual_fn}

    def train_y(self):
        raise NotImplementedError
        # TODO Return all train y in a numpy array from {sample}_{self.norm_fn}_{self.residual_fn}

    def test_x(self):
        raise NotImplementedError
        # TODO Return all test x in a numpy array from {sample}_{self.norm_fn}_{self.residual_fn}

    def test_y(self):
        raise NotImplementedError
        # TODO Return all test y in a numpy array from {sample}_{self.norm_fn}_{self.residual_fn}


class TFLearn_Interface(Interface):
    def train_x(self):
        raise NotImplementedError
        # TODO Return an hdf5 dataset containing all train x from {sample}_{self.norm_fn}_{self.residual_fn}

    def train_y(self):
        raise NotImplementedError
        # TODO Return an hdf5 dataset containing all train y from {sample}_{self.norm_fn}_{self.residual_fn}

    def test_x(self):
        raise NotImplementedError
        # TODO Return an hdf5 dataset containing all test x from {sample}_{self.norm_fn}_{self.residual_fn}

    def test_y(self):
        raise NotImplementedError
        # TODO Return an hdf5 dataset containing all test y from {sample}_{self.norm_fn}_{self.residual_fn}
