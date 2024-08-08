from abc import ABC, abstractmethod
from itertools import combinations
from typing import Literal, get_args

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

from adaptivee.utils import deprecated

SaticMethodTypes = Literal["accuracy", "difference"]


class MixInTargetWeighter(ABC):

    def __init__(self) -> None:
        super().__init__()

    def get_target_weights(
        self, models_preds: np.ndarray, true_y: np.ndarray
    ) -> np.ndarray:
        true_y = true_y.reshape((-1, 1))
        weights = self._get_target_weights(models_preds, true_y)

        return weights

    @abstractmethod
    def _get_target_weights(
        self, models_preds: np.ndarray, true_y: np.ndarray
    ) -> np.ndarray:
        pass


class MixInStaticTargetWeighter(MixInTargetWeighter):

    def __init__(self) -> None:
        super().__init__()

        self.weights = None

    def _get_target_weights(
        self, models_preds: np.ndarray, true_y: np.ndarray
    ) -> np.ndarray:

        if self.weights is None:
            self.weights = self._find_best_weights(models_preds, true_y)

        return self.weights.reshape(1, -1).repeat(
            repeats=true_y.shape[0], axis=0
        )

    @abstractmethod
    def _find_best_weights(
        self, models_preds: np.ndarray, true_y: np.ndarray
    ) -> np.ndarray:
        pass


class SoftMaxWeighter(MixInTargetWeighter):

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def _get_target_weights(
        self, models_preds: np.ndarray, true_y: np.ndarray
    ) -> np.ndarray:

        diffs = np.abs(models_preds) - true_y
        weights = softmax(1 - diffs, axis=1)

        return weights


class OneHotWeighter(MixInTargetWeighter):

    def _get_target_weights(
        self, models_preds: np.ndarray, true_y: np.ndarray
    ) -> np.ndarray:

        diffs = np.abs(models_preds - true_y)
        best_scores = diffs.min(axis=1)

        weights = np.where(diffs == best_scores[:, np.newaxis], 1, 0)
        weights = weights / weights.sum(axis=1).reshape((-1, 1))

        return weights


@deprecated
class StaticGridWeighter(MixInStaticTargetWeighter):

    def __init__(
        self,
        method: SaticMethodTypes = "accuracy",
        grid_points: int = 10,
    ) -> None:
        super().__init__()

        if method not in get_args(SaticMethodTypes):
            raise TypeError(
                f"method should be from the {get_args(SaticMethodTypes)}, got {method} instead."
            )

        self.metohd = method
        self.grid_points = grid_points

    def _find_best_weights(
        self, models_preds: np.ndarray, true_y: np.ndarray
    ) -> np.ndarray:
        n_models = models_preds.shape[1]

        step_size = 1 / self.grid_points
        grid_1d = np.arange(0, 1 + step_size, step_size)

        grid_points = combinations(grid_1d, n_models)

        max_score = 0
        best_weights = None
        for grid_point in grid_points:

            if abs(sum(grid_point) - 1) > 1e-2:
                continue

            score = self._evaluate_weighting(grid_point, models_preds, true_y)
            if score > max_score:
                best_weights = np.array(grid_point)
                max_score = score

        return best_weights

    def _evaluate_weighting(
        self,
        weights: list[float],
        models_preds: np.ndarray,
        true_y: np.ndarray,
    ):
        weights = np.array(weights).reshape((1, -1))
        y_pred = models_preds * weights

        if self.method == "accuracy":
            res = np.mean((y_pred > 0.5).astype(int) == true_y)
        if self.method == "difference":
            res = 1 - np.mean(np.abs(y_pred - true_y))

        return res


class StaticLogisticWeighter(MixInStaticTargetWeighter):

    def __init__(self) -> None:
        super().__init__()

    def _find_best_weights(
        self, models_preds: np.ndarray, true_y: np.ndarray
    ) -> np.ndarray:

        linear_model = LogisticRegression()
        linear_model.fit(models_preds, true_y.reshape(-1))
        weights = linear_model.coef_
        weights = weights / weights.sum()

        return weights


class StaticEqualWeighter(MixInStaticTargetWeighter):

    def _find_best_weights(
        self, models_preds: np.ndarray, true_y: np.ndarray
    ) -> np.ndarray:
        p = models_preds.shape[1]
        weights = np.ones(shape=(p)) / p

        return weights
