import numpy as np
from pytest import fixture
from sklearn.datasets import make_classification


class Utils:

    @staticmethod
    def numpy_arrays_equal(
        a1: np.ndarray, a2: np.ndarray | float, precision: float = 1e-6
    ):
        return all(((a1 - a2) < precision).reshape(-1))


@fixture(scope="session")
def utils():
    return Utils


@fixture(scope="session")
def dataset_1() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(n_samples=1_000, random_state=42)
    return X, y


@fixture(scope="session")
def true_y() -> np.ndarray:
    y = np.array([1, 1, 0, 0])
    return y


@fixture(scope="session")
def onehot_pred_y() -> np.ndarray:
    y_pred = np.array(
        [
            [0.1, 1.0, 0.9],
            [1.0, 0.1, 0.8],
            [0.1, 0.0, 0.3],
            [0.0, 0.0, 0.5],
        ]
    )
    return y_pred


@fixture(scope="session")
def models_pred() -> np.ndarray:
    y_pred = np.array(
        [
            [0.1, 1.0, 0.9],
            [1.0, 0.1, 0.8],
            [0.1, 0.0, 0.3],
            [0.5, 0.0, 0.5],
        ]
    )
    return y_pred


@fixture(scope="session")
def model_one_perfect_pred() -> np.ndarray:
    """first model is perfect for true_y"""
    y_pred = np.array(
        [
            [1.0, 1.0, 0.9],
            [1.0, 0.1, 0.8],
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.5],
        ]
    )
    return y_pred


@fixture(scope="session")
def initial_weights():
    weights = np.array(
        [
            [0.1, 0.0, 0.9],
            [1.0, 0.0, 0.0],
            [0.2, 0.5, 0.3],
            [0.5, 0.0, 0.5],
        ]
    )
    return weights


@fixture(scope="session")
def predicted_weights():
    weights = np.array(
        [
            [0.15, 0.05, 0.8],
            [0.5, 0.2, 0.3],
            [0.1, 0.6, 0.3],
            [0.5, 0.0, 0.5],
        ]
    )
    return weights
