from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
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


class DirectionReweight(MixInReweight):

    def __init__(self, step_size: float = 0.1) -> None:
        super().__init__()
        self.step_size = step_size

    def _get_final_weights(
        self,
        encoder_weights: Tensor,
        initial_weights: Tensor | None = None,
    ) -> Tensor:
        weights = (
            initial_weights
            + (encoder_weights - initial_weights) * self.step_size
        )
        # weights = weights / weights.sum(dim=1)
        return weights


class DirectionConstantReweight(MixInReweight):

    def __init__(self, step_size: float = 0.01) -> None:
        super().__init__()
        self.step_size = step_size

    def _get_final_weights(
        self, encoder_weights: Tensor, initial_weights: Tensor | None = None
    ) -> Tensor:
        weights = initial_weights + np.sign(encoder_weights) * self.step_size
        weights = weights / weights.sum(axis=1).reshape((-1, 1))
        return weights
