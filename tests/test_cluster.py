import numpy as np
from interdim.cluster import cluster_data

def test_kmeans_clustering(small_dataset):
    labels = cluster_data(small_dataset, method='kmeans', n_clusters=3)
    assert len(np.unique(labels)) == 3

def test_dbscan_clustering(small_dataset):
    labels = cluster_data(small_dataset, method='dbscan')
    assert len(np.unique(labels)) >= 1  # DBSCAN may produce any number of clusters

def test_invalid_method(small_dataset):
    try:
        cluster_data(small_dataset, method='invalid_method')
    except ValueError:
        assert True
    else:
        assert False, "Invalid method should raise ValueError"