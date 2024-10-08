import numpy as np

from adaptivee.target_weights import (
    OneHotWeighter,
    SoftMaxWeighter,
    StaticEqualWeighter,
    StaticLogisticWeighter,
)


def test_softmax_weighter_sum_to_1(true_y, models_pred):

    weighter = SoftMaxWeighter()
    weights = weighter.get_target_weights(models_pred, true_y)

    assert all(np.abs(weights.sum(axis=1) - 1) < 1e-4)


def test_softmax_weighter_best_highest(true_y, models_pred):

    weighter = SoftMaxWeighter()
    weights = weighter.get_target_weights(models_pred, true_y)

    diffs = np.abs(models_pred - true_y.reshape((-1, 1)))

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


def test_onehot_sum_to_1(true_y, onehot_pred_y):

    weighter = OneHotWeighter()
    weights = weighter.get_target_weights(onehot_pred_y, true_y)

    assert all(weights.sum(axis=1) == 1)


def test_onehot_equal_share(true_y, onehot_pred_y):

    weighter = OneHotWeighter()
    weights = weighter.get_target_weights(onehot_pred_y, true_y)

    assert weights[3, 0] == 0.5
    assert weights[3, 1] == 0.5

    assert weights[0, 1] == 1
    assert weights[1, 0] == 1
    assert weights[2, 1] == 1
