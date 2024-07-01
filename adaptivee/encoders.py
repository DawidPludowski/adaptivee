from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


class MixInEncoder(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(
        self,
        X: Tensor | np.ndarray | pd.DataFrame,
        y: Tensor | np.ndarray | pd.DataFrame,
    ) -> None:
        pass

    @abstractmethod
    def predict(
        self,
        X: Tensor | np.ndarray | pd.DataFrame,
    ) -> np.ndarray:
        pass

    @property
    def encoder(self):
        return self._encoder


class MixInDeepEncoder(MixInEncoder):

    def __init__(self) -> None:
        pass

    def train(
        self,
        X: Tensor | np.ndarray | pd.DataFrame,
        y: Tensor | np.ndarray | pd.DataFrame,
        n_iter: int = 100,
    ) -> None:

        self.encoder.train()

        X, y = self._convert_dtype(X), self._convert_dtype(y)
        dataloder = self._loader_from_data(X, y)
        self._train(dataloder, n_iter=n_iter)

    @torch.no_grad()
    def predict(
        self,
        X: Tensor | np.ndarray | pd.DataFrame,
    ) -> np.ndarray:
        self.encoder.eval()
        X = self._convert_dtype(X)
        X = X.float()

        y_preds = self.encoder(X)
        return y_preds.numpy()

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

    def _loader_from_data(
        self, X: Tensor, y: Tensor, batch_size: int = 64
    ) -> DataLoader:
        X, y = X.float(), y.float()
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        return dataloader

    def _get_tb_writer(self, name: str) -> SummaryWriter:
        tb_writer = SummaryWriter(
            f"tensorboard/{name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        return tb_writer

    @abstractmethod
    def _train(self, dataloader: DataLoader, n_iter: int) -> None:
        pass

    @property
    def encoder(self):
        return self._encoder


class NLPEncoder(MixInDeepEncoder):

    def __init__(self, shape: list[int]) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        for i in range(len(shape) - 2):
            layers.append(nn.Linear(shape[i], shape[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(shape[-2], shape[-1]))
        layers.append(nn.Softmax())

        self._encoder = nn.Sequential(*layers)

    def _train(self, dataloader: DataLoader, n_iter: int = 10_000) -> None:

        running_loss = 0.0
        last_loss = 0.0

        tb_writer = self._get_tb_writer("NLP")

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.0001)

        for epoch_index in range(n_iter):

            for i, data in enumerate(dataloader):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.encoder(inputs)

                loss = loss_fn(outputs.reshape(-1), labels.reshape(-1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:
                    last_loss = running_loss / 1000
                    tb_x = epoch_index * len(dataloader) + i + 1
                    tb_writer.add_scalar("Loss/train", last_loss, tb_x)
                    running_loss = 0.0


class DummyEncoder(MixInEncoder):

    def __init__(
        self, output_size: int, weights: Optional[np.ndarray] = None
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.weights = weights

    def train(
        self,
        X: Tensor | np.ndarray | pd.DataFrame,
        y: Tensor | np.ndarray | pd.DataFrame,
    ) -> None:
        pass

    def predict(self, X: Tensor | np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.weights is None:
            return (
                np.ones(shape=(X.shape[0], self.output_size))
                / self.output_size
            )
        else:
            weights = self.weights.reshape((1, -1))
            return np.repeat(weights, axis=0, repeats=X.shape[0])
