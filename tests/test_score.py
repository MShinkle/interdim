import numpy as np
from interdim.score import score_clustering

def test_silhouette_score(small_dataset):
    labels = np.random.randint(0, 3, 100)
    score = score_clustering(small_dataset, labels, method='silhouette')
    assert -1 <= score <= 1

def test_calinski_harabasz_score(small_dataset):
    labels = np.random.randint(0, 3, 100)
    score = score_clustering(small_dataset, labels, method='calinski_harabasz')
    assert score >= 0

def test_invalid_method(small_dataset):
    labels = np.random.randint(0, 3, 100)
    try:
        score_clustering(small_dataset, labels, method='invalid_method')
    except ValueError:
        assert True
    else:
        assert False, "Invalid method should raise ValueError"