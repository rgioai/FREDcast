import numpy as np
import h5py

quandl_codes = []    # list of quandl codes; must be identical order every time

hdf5 = h5py.File('FREDcast.hdf5')  # Open the hdf5 file
hdf5.require_group('data')  # Ensure we have the correct group
dset = hdf5.create_dataset('data/sample_raw', shape=(600, len(quandl_codes)), dtype=np.float32)  # Create an empty dataset
# In production, we'll probably use hdf5.require_dataset('data/raw'), after creating the empty dataset only once.

for i in range(len(quandl_codes)):
    quandl_code = quandl_codes[i]
    base_array = np.empty((600,)).fill(np.nan)  # Create an empty array of 50 years * 12 mos and fill each entry with nan
    quandl_values = np.array([])  # Get quandl values from quandl_code
    # If freq is monthly:
    return_array = np.array([])  # Map the values from quandl_values onto base_array, creating return_array
    # If freq is > monthly, average the weekly or daily values to get a monthly value
    # If freq is < monthly, spread the values across multiple months (4 for quarterly and 12 for annual)
    dset[i] = return_array  # The slicing here is on the wrong axis (row instead of column), but the fix is simple

hdf5.close()
