from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from openml import datasets, study, tasks
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from bin.data.args import get_download_args as get_args


def _get_generic_preprocessing() -> Pipeline:
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one-hot",
                OneHotEncoder(
                    sparse_output=False, handle_unknown="ignore", drop="first"
                ),
            ),
        ]
    )

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    pipeline = Pipeline(
        [
            (
                "transformers",
                make_column_transformer(
                    (
                        cat_pipeline,
                        make_column_selector(
                            dtype_include=("object", "category")
                        ),
                    ),
                    (
                        num_pipeline,
                        make_column_selector(dtype_include=np.number),
                    ),
                ),
            )
        ]
    )

    return pipeline


def _ensure_last_target(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    if df.columns[-1] == target_name:
        return df

    colnames = df.columns.to_list()

    target_idx = colnames.index(target_name)
    order = [i for i in range(len(colnames))]

    order[target_idx], order[len(colnames) - 1] = (
        order[len(colnames) - 1],
        order[target_idx],
    )

    reodered_columns = [colnames[i] for i in order]

    df = df[reodered_columns]
    return df


def get_tasks() -> list[int]:
    tasks_ids = study.get_suite(99).tasks
    return tasks_ids


def download_tasks(
    tasks_: list[int],
    download_dst: str | Path,
    max_downloads: int = 100,
) -> None:

    download_dst = Path(download_dst)
    download_dst.mkdir(exist_ok=True, parents=True)

    downloaded_cnt = 0

    for id_ in tasks_:
        task = tasks.get_task(id_, download_splits=False)

        dataset = datasets.get_dataset(
            task.dataset_id,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )

        task_name = dataset.name

        logger.info(
            f"downloading task: {task_name}, {downloaded_cnt}/{max_downloads}"
        )

        target_name = task.target_name
        data = dataset.get_data()[0]

        data_size = data.memory_usage(index=True).sum() / 1024 // 1024
        if data_size > 200:
            logger.warning(f"Dataset skipped due to its size: {data_size} MB")
            continue

        n_classes = np.unique(data[target_name]).shape[0]
        if n_classes != 2:
            logger.warning(f"Dataset skipped due to class number: {n_classes}")
            continue

        logger.info(f"data size: {data_size} MB")

        data = _ensure_last_target(data, target_name)

        X, y = data.iloc[:, :-1], data.iloc[:, -1]

        pipeline = _get_generic_preprocessing()
        X = pipeline.fit_transform(X)
        y = LabelEncoder().fit_transform(y)

        np.savez(download_dst / f"{task_name}.npz", X=X, y=y)

        downloaded_cnt += 1

        if downloaded_cnt == max_downloads:
            logger.info(
                f"Stopped due to max_donwloads parameter={max_downloads}"
            )
            break

    logger.info(f"Done. Total number of tasks: {downloaded_cnt}")


def main() -> None:

    args = get_args()
    download_dst = args.outdir
    max_downloads = args.max_downloads

    print(args)

    logger.info("Searching for CC18 AutoML tasks...")
    tasks_ = get_tasks()

    logger.info(f"Downloading data to location: {download_dst} ...")
    download_tasks(tasks_, download_dst, max_downloads=max_downloads)


if __name__ == "__main__":
    main()
