from pathlib import Path
import numpy as np
import gzip


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

        with gzip.open(datapath / "X.npy.gz", "rb") as f:
            X = np.load(f, allow_pickle=True)
            X = X.astype(float)
        with gzip.open(datapath / "y.npy.gz", "rb") as f:
            y = np.load(f, allow_pickle=True)
            y = y.astype(float)

        np.savez(output_data_path / f"{datapath.stem}.npz", X=X, y=y)


if __name__ == "__main__":
    main()
