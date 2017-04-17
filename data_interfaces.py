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
    def __init__(self):
        # TODO If split_data.hdf5 does not exist, prompt the user
        # if yes, create it; otherwise quit
        raise NotImplementedError

    def train_x(self, sample=False):
        raise NotImplementedError

    def train_y(self, sample=False):
        raise NotImplementedError

    def test_x(self, sample=False):
        raise NotImplementedError

    def test_y(self, sample=False):
        raise NotImplementedError

    def get_data(self, sample=False):
        return self.train_x(sample), self.train_y(sample), self.test_x(sample), self.test_y(sample)


class SKLearn_Interface(Interface):
    def train_x(self, sample=False):
        raise NotImplementedError
        # TODO Return all train x in a numpy array

    def train_y(self, sample=False):
        raise NotImplementedError
        # TODO Return all train y in a numpy array

    def test_x(self, sample=False):
        raise NotImplementedError
        # TODO Return all test x in a numpy array

    def test_y(self, sample=False):
        raise NotImplementedError
        # TODO Return all test y in a numpy array


class TFLearn_Interface(Interface):
    def train_x(self, sample=False):
        raise NotImplementedError
        # TODO Return an hdf5 dataset containing all train x

    def train_y(self, sample=False):
        raise NotImplementedError
        # TODO Return an hdf5 dataset containing all train y

    def test_x(self, sample=False):
        raise NotImplementedError
        # TODO Return an hdf5 dataset containing all test x

    def test_y(self, sample=False):
        raise NotImplementedError
        # TODO Return an hdf5 dataset containing all test y
