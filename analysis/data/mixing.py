import numpy as np


def change_position(
    data: tuple[np.ndarray, np.ndarray], loc: np.ndarray | list
) -> tuple[np.ndarray, np.ndarray]:

    X, y = data

    if isinstance(loc, list):
        loc = np.array(loc)

    _, p = data[0].shape
    assert loc.reshape(-1).shape[0] == p

    loc = loc.reshape((1, p))
    X += loc

    return (X, y)


def change_variance(
    data: tuple[np.ndarray, np.ndarray], variance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    X, y = data

    _, p = data.shape
    assert variance.reshape(-1).shape[0] == p

    variance = variance.reshape((1, p))
    X /= X.std(axis=0)
    X *= variance

    return (X, y)


def blur_data(
    data: tuple[np.ndarray, np.ndarray], magnitude: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:

    X, y = data

    noise = np.random.normal(size=data[0].shape, scale=magnitude)
    X += noise

    return (X, y)


def mix_data(
    data1: tuple[np.ndarray, np.ndarray],
    data2: tuple[np.ndarray, np.ndarray],
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray]:

    X = np.concatenate([data1[0], data2[0]])
    y = np.concatenate([data1[1], data2[1]])

    if shuffle:
        order = np.random.permutation(X.shape[0])
        X, y = X[order, :], y[order]

    return (X, y)
