import numpy as np

def get_feature_vector(norm_fn, residual_fn=None):
    # TODO Get the feature vector of the most recent data (top row of the data table)
    pass

def generate_all_results(trained_classifier, norm_fn, residual_fn):
    clf = trained_classifier
    feature_vector = get_feature_vector(norm_fn, residual_fn)
    original_prediction = clf.predict(feature_vector)
    
    all_results = np.empty((len(feature_vector), 2), dtype=np.float32)
    for feature_index in range(len(feature_vector)):
        feature_results = np.empty((30,), dtype=np.float32)
        for scale in range(30+1):
            if scale >= 15:
                factor = ((scale + 1) * 0.01) + 0.85
            else:
                factor = (scale * 0.01) + 0.85
            test_vector = feature_vector
            test_vector[feature_index] = feature_vector[feature_index] * scale
            feature_results[scale] = clf.predict(test_vector)
        all_results[feature_index][0] = np.mean(feature_results)  # TODO Verify numpy API
        all_results[feature_index][1] = np.std(feature_results)  # TODO Verify numpy API
    return all_results


def generate_significance(all_results, confidence_interval=0.95):
    all_significance = np.empty((len(all_results, 3)), dtype=np.float32)

    z = 1.96  # TODO Support CI of 0.90, 0.95, 0.99 at least; more if easy

    for i in range(len(all_results)):
        all_significance[i][0] = all_results[i][0] - (z * all_results[i][1])
        all_significance[i][1] = all_results[i][0]
        all_significance[i][2] = all_results[i][0] + (z * all_results[i][1])

    return all_significance


def generate_winners_losers(all_significance):
    winners = []
    # TODO fill winners with the names of the five features that
    # have the highest means (entry 1)
    # and also have a CI above zero (entry 0 is positive)

    losers = []
    # TODO fill losers with the names of the five features that
    # have the lowest means (entry 1)
    # and also have a CI below zero (entry 2 is negative)

    return winners, losers
