import numpy as np
from pytest import fixture
from sklearn.datasets import make_classification


@fixture(scope="session")
def dataset_1() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(n_samples=1_000, random_state=42)
    return X, y
