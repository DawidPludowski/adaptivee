from itertools import combinations, combinations_with_replacement, permutations
from typing import Literal, Optional, get_args

import numpy as np

Distribution = Literal["normal", "uniform"]


def create_linear(
    n: int = 1000,
    p: int = 2,
    betas: Optional[list[float] | np.ndarray] = None,
    distribution: Distribution = "normal",
) -> tuple[np.ndarray, np.ndarray]:

    if betas is None:
        betas = np.random.normal(size=[p, 1])
    else:
        betas = betas.reshape((p, 1))

    X = _sample_data(n, p, distribution)
    y_prob = _sigmoid(X @ betas)

    y = (0.5 < y_prob).astype(int)
    X = _scale(X)

    return X, y


def create_cubes(
    n: int = 1000,
    p: int = 2,
    n_cubes: int = 3,
    distribution: Distribution = "uniform",
) -> tuple[np.ndarray, np.ndarray]:

    assert n_cubes % 2 == 1

    X = _sample_data(n, p, distribution)
    X = _scale(X)
    X_min, X_max = X.min(), X.max()

    y = np.zeros(shape=n)

    cube_size = (X_max - X_min) / n_cubes
    lower_boundaries_ = np.arange(X_min, X_max, cube_size)

    lower_boundaries = set(combinations_with_replacement(lower_boundaries_, p))
    print(len(lower_boundaries))
    all_boundaries = []

    for boundary in lower_boundaries:
        all_boundaries += permutations(boundary)

    all_boundaries = set(all_boundaries)
    print(len(all_boundaries))

    for i in range(p):
        all_boundaries = sorted(all_boundaries, key=lambda x: x[i])

    for idx, lower_boundary in enumerate(all_boundaries):
        within_boundary_idx = (
            ((X >= lower_boundary) * (X < (lower_boundary + cube_size)))
            .astype(int)
            .prod(axis=1)
            .astype(bool)
        )
        y[within_boundary_idx] = idx % 2

    return X, y


def create_circles(
    n: int = 1000,
    p: int = 2,
    n_circles: int = 2,
    distribution: Distribution = "normal",
) -> tuple[np.ndarray, np.ndarray]:

    X = _sample_data(n, p, distribution)
    radius = np.sqrt((X**2).sum(axis=1))
    radius_min, radius_max = radius.min(), radius.max()

    radius_size_diff = (radius_max - radius_min) / n_circles
    radius_sizes = np.arange(radius_min, radius_max, radius_size_diff)

    y = np.zeros(shape=n)

    for idx, radius_size in enumerate(radius_sizes):
        y[
            (radius >= radius_size)
            & (radius_size < (radius_size + radius_size_diff))
        ] = (idx % 2)

    return X, y


def create_normal_distribution(
    n: int = 1000,
    p: int = 2,
    locs: Optional[tuple[list | np.ndarray, list | np.ndarray]] = None,
    distribution: Distribution = "normal",
) -> np.ndarray:

    X_1 = _sample_data(n // 2, p, distribution)
    X_2 = _sample_data(n // 2, p, distribution)

    if locs is None:
        loc_1, loc_2 = np.random.normal(size=p, loc=-1), np.random.normal(
            size=p, loc=1
        )
    else:
        loc_1, loc_2 = np.array(locs[0]), np.array(locs[1])

    X_1 = X_1 + loc_1
    X_2 = X_2 + loc_2

    X = np.vstack((X_1, X_2))
    y = np.concatenate((np.ones(shape=n // 2), np.zeros(shape=n // 2)))

    permutation = np.random.permutation(X.shape[0])
    X = X[permutation, :]
    y = y[permutation]

    return X, y


def _scale(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0)) / X.std(axis=0)


def _sample_data(n: int, p: int, distribution: Distribution) -> np.ndarray:

    assert distribution in get_args(Distribution)

    if distribution == "normal":
        return np.random.normal(size=(n, p))
    elif distribution == "uniform":
        return np.random.uniform(size=(n, p))


def _sigmoid(X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-X))
