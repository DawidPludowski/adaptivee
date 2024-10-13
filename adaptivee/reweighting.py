from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar
from torch import Tensor


class MixInReweight(ABC):

    def __init__(self) -> None:
        super().__init__()

    def get_final_weights(
        self,
        encoder_weights: Tensor,
        initial_weights: Optional[Tensor] = None,
    ) -> np.ndarray:

        final_weights = self._get_final_weights(
            encoder_weights, initial_weights
        )
        return final_weights

    @abstractmethod
    def optimize_hp(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        static_weights: np.ndarray,
        encoder_weights: np.ndarray,
    ) -> None:
        pass

    @abstractmethod
    def _get_final_weights(
        self,
        encoder_weights: Tensor,
        initial_weights: Optional[Tensor] = None,
    ) -> Tensor:
        pass


class SimpleReweight(MixInReweight):

    def __init__(self) -> None:
        super().__init__()

    def _get_final_weights(
        self,
        encoder_weights: Tensor,
        initial_weights: Optional[Tensor] = None,
    ) -> Tensor:
        return encoder_weights

    def optimize_hp(self, y_true, y_pred, static_weights, encoder_weights):
        pass


class DirectionReweight(MixInReweight):

    def __init__(self, step_size: float = 0.1) -> None:
        super().__init__()
        self.step_size = step_size

    def _get_final_weights(
        self,
        encoder_weights: Tensor,
        initial_weights: Tensor | None = None,
    ) -> Tensor:
        weights = self._fun(initial_weights, encoder_weights, self.step_size)
        return weights

    def _fun(self, w1, w2, alpha):
        return w1 + (w2 - w1) * alpha

    def optimize_hp(self, y_true, y_pred, static_weights, encoder_weights):

        fun = lambda alpha: np.mean(
            (
                y_true
                - (
                    y_pred
                    @ (
                        static_weights
                        + (encoder_weights - static_weights) * alpha
                    )
                )
            )
            ** 2
        )

        alpha = minimize_scalar(fun, bounds=(0, 1)).x
        self.step_size = alpha


class DirectionConstantReweight(MixInReweight):

    def __init__(self, step_size: float = 0.1) -> None:
        super().__init__()
        self.step_size = step_size

    def _get_final_weights(
        self, encoder_weights: Tensor, initial_weights: Tensor | None = None
    ) -> Tensor:
        weights = initial_weights + np.sign(encoder_weights) * self.step_size
        weights = weights / weights.sum(axis=1).reshape((-1, 1))
        return weights

    def optimize_hp(self, y_true, y_pred, static_weights, encoder_weights):
        pass
