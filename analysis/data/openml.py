from pathlib import Path
import numpy as np


def get_data(path: str):
    path = Path(path)

    train_path = path / "train"
    test_path = path / "test"

    for data_path in train_path.glob("*.npz"):
        dataname = data_path.stem

        train_data = np.load(train_path / f"{dataname}.npz")
        test_data = np.load(test_path / f"{dataname}.npz")

        yield train_data, test_data, dataname
