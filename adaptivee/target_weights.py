from abc import ABC, abstractmethod

import numpy as np
from scipy.special import softmax


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
