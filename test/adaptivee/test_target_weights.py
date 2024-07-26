import numpy as np

from adaptivee.target_weights import (
    SoftMaxWeighter,
    StaticEqualWeighter,
    StaticLogisticWeighter,
)


def test_softmax_weighter_sum_to_1(true_y, models_pred):

    weighter = SoftMaxWeighter()
    weights = weighter.get_target_weights(models_pred, true_y)

    assert all(weights.sum(axis=1) == 1)


def test_softmax_weighter_best_highest(true_y, models_pred):

    weighter = SoftMaxWeighter()
    weights = weighter.get_target_weights(models_pred, true_y)

    diffs = models_pred - true_y.reshape((-1, 1))
    best_pred = np.argmin(diffs, axis=1)
    highest_weights = np.argmax(weights, axis=1)

    assert all(best_pred == highest_weights)


def test_static_logistic_coeffs(model_one_perfect_pred, true_y):

    weighter = StaticLogisticWeighter()
    _ = weighter.get_target_weights(model_one_perfect_pred, true_y)

    weights = weighter.weights

    assert weights.argmax() == 0


def test_static_equal_weighter(true_y, models_pred, utils):

    weighter = StaticEqualWeighter()
    _ = weighter.get_target_weights(models_pred, true_y)

    weights = weighter.weights

    assert utils.numpy_arrays_equal(weights, 1 / 3)


def test_weights_size(true_y, models_pred):

    weighter = SoftMaxWeighter()
    weights = weighter.get_target_weights(models_pred, true_y)

    assert len(weights.shape) == 2
    assert weights.shape[0] == models_pred.shape[0]
    assert weights.shape[1] == models_pred.shape[1]
