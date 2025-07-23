from pathlib import Path
import numpy as np
import gzip
from bin.utils import get_generic_preprocessing
import pandas as pd
from loguru import logger


def main() -> None:
    cwd = Path.cwd()
    if cwd.stem == "adaptivee":
        path = cwd / "resources" / "tabzilla"
    elif cwd.stem == "tabzilla":
        path = cwd
    else:
        raise ValueError(
            "Incorrect context path (should be either root o repo or tabzilla directory)"
        )

    input_data_path = path / "datasets"
    output_data_path = path / "datasets_restructured"

    output_data_path.mkdir(parents=True, exist_ok=True)

    for datapath in (input_data_path).glob("*"):
        logger.info(f"Start - {datapath.name}")

        with gzip.open(datapath / "X.npy.gz", "rb") as f:
            X = np.load(f, allow_pickle=True)

            # missing values hotfix
            X = X.astype(float)
            pipeline = get_generic_preprocessing()
            X = pipeline.fit_transform(pd.DataFrame(X))

        with gzip.open(datapath / "y.npy.gz", "rb") as f:
            y = np.load(f, allow_pickle=True)
            y = y.astype(float)

        np.savez(output_data_path / f"{datapath.stem}.npz", X=X, y=y)


if __name__ == "__main__":
    main()
