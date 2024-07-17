from abc import ABC, abstractmethod
from itertools import combinations
from typing import Literal, get_args

import numpy as np
from scipy.special import softmax

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


class SoftMaxWeighter(MixInTargetWeighter):

    def __init__(self, regularization_term: float = 0) -> None:
        super().__init__()
        self.C = regularization_term

    def _get_target_weights(
        self, models_preds: np.ndarray, true_y: np.ndarray
    ) -> np.ndarray:

        diffs = models_preds - true_y
        weights = softmax(1 - diffs, axis=1)

        return weights


class StaticGridWeighter(MixInTargetWeighter):

    def __init__(
        self, grid_points: int = 10, method: SaticMethodTypes = "accuracy"
    ) -> None:
        super().__init__()
        if method not in get_args(SaticMethodTypes):
            raise TypeError(
                f"method should be from the {get_args(SaticMethodTypes)}, got {method} instead."
            )

        self.grid_points = grid_points
        self.method = method
        self.weights = None

    def _get_target_weights(
        self, models_preds: np.ndarray, true_y: np.ndarray
    ) -> np.ndarray:

        if self.weights is None:
            self.weights = self._find_best_weights(models_preds, true_y)

        return self.weights.reshape(1, -1).repeat(
            repeats=true_y.shape[0], axis=0
        )

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
