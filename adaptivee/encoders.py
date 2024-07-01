from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn


class Encoder(ABC):

    def __init__(self) -> None:
        pass

    def train(
        self,
        X: Tensor | np.ndarray | pd.DataFrame,
        y: Tensor | np.ndarray | pd.DataFrame,
    ) -> None:

        X, y = self._convert_dtype(X), self._convert_dtype(y)

    def _convert_dtype(
        self, data: Tensor | np.ndarray | pd.DataFrame
    ) -> Tensor:

        if isinstance(data, Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, pd.DataFrame):
            return torch.from_numpy(data.to_numpy())
        else:
            raise TypeError(
                f"Expected type from Tensor | np.ndarray | pd.DataFrame. Got {type(data)} instead."
            )

    @abstractmethod
    def _train(self, X: Tensor, y: Tensor):
        pass

    @property
    def encoder(self):
        return self._encoder


class NLPEncoder(Encoder):

    def __init__(self, shape: list[int]) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        for i in range(len(shape) - 1):
            layers.append(nn.Linear(shape[i], shape[i + 1]))
            layers.append(nn.ReLU())

        self._encoder = nn.Sequential(*layers)

    def _train(self, X: Tensor, y: Tensor):
        pass
