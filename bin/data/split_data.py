from pathlib import Path
from bin.data.args import get_split_args as get_args
from sklearn.model_selection import train_test_split
from shutil import copy
import numpy as np


def inner_split(
    paths: list[Path], outpath: Path, train_size: float, seed: int
) -> None:

    (outpath / "train").mkdir(exist_ok=True, parents=True)
    (outpath / "test").mkdir(exist_ok=True, parents=True)

    for path in paths:
        data = np.load(path)
        X, y = data["X"], data["y"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=seed
        )

        np.savez(outpath / "train" / path.name, X=X_train, y=y_train)
        np.savez(outpath / "test" / path.name, X=X_test, y=y_test)


def outer_split(
    paths: list[Path], outpath: Path, train_size: float, seed: int
) -> None:

    (outpath / "train").mkdir(exist_ok=True, parents=True)
    (outpath / "test").mkdir(exist_ok=True, parents=True)

    train_files, test_files = train_test_split(
        paths,
        train_size=train_size,
        random_state=seed,
    )

    # train
    out_subpath = outpath / "train"
    out_subpath.mkdir(exist_ok=True, parents=True)
    for path in train_files:
        copy(path, out_subpath / path.name)

    # test
    out_subpath = outpath / "test"
    out_subpath.mkdir(exist_ok=True, parents=True)
    for path in test_files:
        copy(path, out_subpath / path.name)


def main():
    args = get_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    dataset_files = list(input_path.glob("*.npz"))

    inference_files, encoder_files = train_test_split(
        dataset_files, train_size=args.inference_frac, random_state=args.seed
    )

    if args.inference_outer_split:
        outer_split(
            inference_files,
            output_path / "inference",
            args.inference_train_frac,
            args.seed,
        )
    else:
        inner_split(
            inference_files,
            output_path / "inference",
            args.inference_train_frac,
            args.seed,
        )

    if args.encoder_outer_split:
        outer_split(
            encoder_files,
            output_path / "encoder",
            args.encoder_train_frac,
            args.seed,
        )
    else:
        inner_split(
            encoder_files,
            output_path / "encoder",
            args.encoder_train_frac,
            args.seed,
        )


if __name__ == "__main__":
    main()
