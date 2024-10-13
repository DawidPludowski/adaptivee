from abc import ABC, abstractmethod
from itertools import combinations
from typing import Literal, get_args

import numpy as np
import pandas as pd
from autogluon.core.models.ensemble.weighted_ensemble_model import (
    WeightedEnsembleModel,
)
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

from adaptivee.utils import deprecated

SaticMethodTypes = Literal["accuracy", "difference"]


class MixInTargetWeighter(ABC):

    def __init__(self) -> None:
        super().__init__()

    def get_target_weights(
        self, models_preds: np.ndarray, true_y: np.ndarray | pd.Series
    ) -> np.ndarray:

        if isinstance(true_y, pd.Series) or isinstance(true_y, pd.DataFrame):
            true_y = true_y.to_numpy()

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

    def __init__(self, alpha: float = 0.9) -> None:
        self.alpha = alpha
        super().__init__()

    def _get_target_weights(
        self, models_preds: np.ndarray, true_y: np.ndarray
    ) -> np.ndarray:

        diffs = np.abs(models_preds - true_y)
        weights = softmax((1 - diffs) * self.alpha, axis=1)

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


class StaticGridWeighter(MixInStaticTargetWeighter):

    def __init__(
        self,
        method: SaticMethodTypes = "accuracy",
        precision: int = 25,
    ) -> None:
        super().__init__()

        if method not in get_args(SaticMethodTypes):
            raise TypeError(
                f"method should be from the {get_args(SaticMethodTypes)}, got {method} instead."
            )

        self.method = method
        self.precision = precision

    def _find_best_weights(
        self, models_preds: np.ndarray, true_y: np.ndarray
    ) -> np.ndarray:
        n_models = models_preds.shape[1]

        grid_points = self._get_grid_points(n_models, self.precision)

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

        if isinstance(true_y, pd.Series) or isinstance(true_y, pd.DataFrame):
            true_y = true_y.to_numpy()

        weights = np.array(weights).reshape((1, -1))
        y_pred = models_preds * weights

        if self.method == "accuracy":
            res = np.mean(
                (y_pred > 0.5).astype(int) == true_y.reshape((-1, 1))
            )
        if self.method == "difference":
            res = 1 - np.mean(np.abs(y_pred - true_y.reshape((-1, 1))))

        return res

    def _get_grid_points(
        self,
        num_elements,
        precision,
        current_index=0,
        current_sum=0,
        current_combination=[],
    ):

        if current_index == num_elements:
            if current_sum == precision:
                return [current_combination]
            return []

        combinations_ = []
        for value in range(precision + 1):
            new_sum = current_sum + value
            if new_sum <= precision:
                new_combination = current_combination + [value / precision]
                combinations_.extend(
                    self._get_grid_points(
                        num_elements,
                        precision,
                        current_index + 1,
                        new_sum,
                        new_combination,
                    )
                )

        return combinations_


@deprecated
class StaticLogisticWeighter(MixInStaticTargetWeighter):

    def __init__(self) -> None:
        super().__init__()

    def _find_best_weights(
        self, models_preds: np.ndarray, true_y: np.ndarray
    ) -> np.ndarray:

        linear_model = LogisticRegression(fit_intercept=False)
        linear_model.fit(models_preds, true_y.reshape(-1))
        weights = linear_model.coef_

        return weights


class StaticEqualWeighter(MixInStaticTargetWeighter):

    def _find_best_weights(
        self, models_preds: np.ndarray, true_y: np.ndarray
    ) -> np.ndarray:
        p = models_preds.shape[1]
        weights = np.ones(shape=(p)) / p

        return weights


class StaticFixedWeights(MixInStaticTargetWeighter):

    def __init__(self, weights: np.ndarray) -> None:
        super().__init__()
        self.weights = weights

    def _find_best_weights(
        self, models_preds: np.ndarray, true_y: np.ndarray
    ) -> np.ndarray:
        return self.weights
