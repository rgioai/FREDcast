import numpy as np
import h5py


def get_feature_vector(norm_fn, residual_fn=None):
    hdf5 = h5py.File('split_data.hdf5')
    if residual_fn is None:
        return np.asarray(hdf5[str(norm_fn) + '/test_x'])[-1, :]
    else:
        return np.asarray(hdf5[str(norm_fn) + '_' + str(residual_fn) + '/test_x'])[-1, :]


def generate_all_results(trained_classifier, norm_fn, residual_fn):
    clf = trained_classifier
    feature_vector = get_feature_vector(norm_fn, residual_fn)
    original_prediction = clf.predict(feature_vector)

    all_results = np.empty((len(feature_vector), 2), dtype=np.float32)
    for feature_index in range(len(feature_vector)):
        feature_results = np.empty((30,), dtype=np.float32)
        for scale in range(30 + 1):
            if scale >= 15:
                factor = ((scale + 1) * 0.01) + 0.85
            else:
                factor = (scale * 0.01) + 0.85
            test_vector = feature_vector
            test_vector[feature_index] = feature_vector[feature_index] * scale
            feature_results[scale] = np.linalg.norm(clf.predict(
                test_vector) - original_prediction)
            # TODO this will be a vector.  Get it's magnitude to make it a scalar.
            # Am I saving it to the correct location in memory?
            ## The array size wouldn't support the two-line solution you had, but it will support the one-liner
            ## feature_results[scale] = np.linalg.norm(feature_results[scale])

        all_results[feature_index][0] = np.mean(feature_results)  # TODO Verify numpy API
        all_results[feature_index][1] = np.std(feature_results)  # TODO Verify numpy API
        # These should return what we want unless there are any NAN values. Then we should use nanmean and nanstd.
        ## Assume there won't be any nan values, they should have been cleaned prior.
    return all_results


def generate_significance(all_results, confidence_interval=0.95):
    all_significance = np.empty((len(all_results, 3)), dtype=np.float32)

    # TODO Support CI of 0.90, 0.95, 0.99 at least; more if easy
    # Not sure what needs to be done here. Are we replacing z with confidence_interval, or something else?
    ## z is the points on the standard normal distribution such that the area under the curve between -z and z is equal
    ## to the confidence interval.  Basic stats uses the above three values mostly, referenced from a 
    ## standard table (http://www.stat.ufl.edu/~athienit/Tables/Ztable.pdf).  That said, it is just an integral,
    ## something that python ought to be able to do.  For example:
    if confidence_interval == 0.90:
        z = ??
    if confidence_interval == 0.95:
        z = 1.96
    elif confidence_interval == 0.99:
        z = ??
    else:
        z = ?? # actually do the full calculation

    for i in range(len(all_results)):
        all_significance[i][0] = all_results[i][0] - (z * all_results[i][1])
        all_significance[i][1] = all_results[i][0]
        all_significance[i][2] = all_results[i][0] + (z * all_results[i][1])

    return all_significance


def generate_winners_losers(all_significance):
    hdf5 = h5py.File('admin.hdf5')
    names = np.asarray(hdf5['codes'])

    winners = []
    # TODO Should be sorting on entry 1 (means), need verified
    ## Correct
    sorted_all = np.sort(all_significance, axis=1)[::-1]
    indices = np.argsort(all_significance, axis=1)[::-1]

    for i in range(len(sorted_all)):
        if sorted_all[i][0] > 0:
            winners.append(names[indices[i]])

    losers = []
    sorted_all = np.sort(all_significance, axis=1)
    indices = np.argsort(all_significance, axis=1)

    for i in range(len(sorted_all)):
        if sorted_all[i][2] < 0:
            losers.append(names[indices[i]])

    return winners, losers
