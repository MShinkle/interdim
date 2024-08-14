import pytest
import numpy as np

@pytest.fixture
def small_dataset():
    return np.random.rand(100, 10)

@pytest.fixture
def labeled_dataset():
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 3, 100)
    return X, y