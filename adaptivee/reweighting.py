from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from torch import Tensor


class MixInReweight(ABC):

    def __init__(self) -> None:
        super().__init__()

    def get_final_weights(
        self,
        encoder_weights: Tensor,
        initial_weights: Optional[Tensor] = None,
        regularization_term: Optional[float] = None,
    ) -> np.ndarray:

        final_weights = self._get_final_weights(
            encoder_weights, initial_weights, regularization_term
        )
        return final_weights.numpy()

    @abstractmethod
    def _get_final_weights(
        self,
        encoder_weights: Tensor,
        initial_weights: Optional[Tensor] = None,
        regularization_term: Optional[float] = None,
    ) -> Tensor:
        pass


class SimpleReweight(MixInReweight):

    def __init__(self) -> None:
        super().__init__()

    def _get_final_weights(
        self,
        encoder_weights: Tensor,
        initial_weights: Optional[Tensor] = None,
        regularization_term: Optional[float] = None,
    ) -> Tensor:
        return encoder_weights
