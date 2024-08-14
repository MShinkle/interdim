import numpy as np
from interdim.reduce import reduce_dimensionality

def test_pca_reduction(small_dataset):
    reduced = reduce_dimensionality(small_dataset, method='pca', n_components=2)
    assert reduced.shape == (100, 2)

def test_tsne_reduction(small_dataset):
    reduced = reduce_dimensionality(small_dataset, method='tsne', n_components=3)
    assert reduced.shape == (100, 3)

def test_invalid_method(small_dataset):
    try:
        reduce_dimensionality(small_dataset, method='invalid_method')
    except ValueError:
        assert True
    else:
        assert False, "Invalid method should raise ValueError"