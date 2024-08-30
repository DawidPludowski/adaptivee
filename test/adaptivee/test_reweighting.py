from adaptivee.reweighting import (
    DirectionConstantReweight,
    DirectionReweight,
    SimpleReweight,
)


def test_simple_reqeight_does_not_change_output(
    predicted_weights, initial_weights, utils
):
    reweighter = SimpleReweight()
    final_weights = reweighter.get_final_weights(
        predicted_weights, initial_weights
    )
    assert utils.numpy_arrays_equal(predicted_weights, final_weights)


def test_direction_reweight_return_sum_to_1(
    predicted_weights, initial_weights, utils
):

    reweighter = DirectionReweight(step_size=0.1)
    final_weights = reweighter.get_final_weights(
        predicted_weights, initial_weights
    )

    assert utils.numpy_arrays_equal(final_weights.sum(axis=1), 1)


def test_direction_reweight_collaps_to_simple_reweight(
    predicted_weights, initial_weights, utils
):

    reweighter = SimpleReweight()
    final_weights_simple = reweighter.get_final_weights(
        predicted_weights, initial_weights
    )

    reweighter = DirectionReweight(step_size=1)
    final_weights_direction = reweighter.get_final_weights(
        predicted_weights, initial_weights
    )

    assert utils.numpy_arrays_equal(
        final_weights_simple, final_weights_direction
    )


def test_direciton_reweight_counts_properly(
    predicted_weights, initial_weights, utils
):

    reweighter = DirectionReweight(step_size=0.1)
    final_weights = reweighter.get_final_weights(
        predicted_weights, initial_weights
    )

    expected_weights = (
        initial_weights + (predicted_weights - initial_weights) * 0.1
    )

    assert utils.numpy_arrays_equal(final_weights, expected_weights)


def test_direction_constant_sum_to_1(
    predicted_weights, initial_weights, utils
):

    reweighter = DirectionConstantReweight(step_size=0.1)
    final_weights = reweighter.get_final_weights(
        predicted_weights, initial_weights
    )

    assert utils.numpy_arrays_equal(final_weights.sum(axis=1), 1)
