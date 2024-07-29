import random
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from openml import datasets, tasks
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ADDITIONAL_CONDITIONS: list[callable] = [
    lambda task: task["NumberOfInstances"] < 10_000,
    lambda task: task["NumberOfMissingValues"] == 0,
    lambda task: task["NumberOfInstances"] > 100,
    lambda task: task["task_type"] == "Supervised Classification",
    lambda task: task["NumberOfSymbolicFeatures"] < 3,
]


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


def get_tasks() -> dict[str, any]:
    classification_tasks = tasks.list_tasks(
        task_type=tasks.TaskType.SUPERVISED_CLASSIFICATION
    )
    return classification_tasks


def filter_tasks(
    classification_tasks: dict[str, any], additional_conditions: callable = {}
) -> list[dict[str, any]]:

    tasks_ = []

    for task in classification_tasks.values():
        if task.get("NumberOfClasses") is None:
            continue
        if task["NumberOfInstances"] <= task["NumberOfFeatures"]:
            continue
        if task["NumberOfClasses"] != 2:
            continue

        for condition in additional_conditions:
            if not condition(task):
                continue

        tasks_.append(task)

    return tasks_


def download_tasks(
    tasks_: list[dict[str, any]],
    download_dst: str | Path,
    max_downloads: int = 100,
    shuffle: bool = True,
    seed: int = 42,
) -> None:

    download_dst = Path(download_dst)
    download_dst.mkdir(exist_ok=True)

    if shuffle:
        random.seed(seed)
        random.shuffle(tasks_)

    downloaded_cnt = 0

    for task in tasks_:
        task_name = task["name"]
        logger.info(
            f"downloading task: {task_name}, {downloaded_cnt}/{max_downloads}"
        )

        target_name = task["target_feature"]
        data = datasets.get_dataset(
            task["did"],
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        ).get_data()[0]

        data_size = data.memory_usage(index=True).sum() / 1024 // 1024
        if data_size > 10:
            logger.warning(f"Dataset skipped due to its size: {data_size}")
            continue

        logger.info(f"data size: {data_size} MB")

        data = _ensure_last_target(data, target_name)

        X, y = data.iloc[:, :-1], data.iloc[:, -1]

        pipeline = _get_generic_preprocessing()
        X = pipeline.fit_transform(X)

        df = pd.DataFrame(data=X)
        df["target"] = y

        df.to_csv(download_dst / f"{task_name}.csv", index=False)

        downloaded_cnt += 1

        if downloaded_cnt == max_downloads:
            logger.info(
                f"Stopped due to max_donwloads parameter={max_downloads}"
            )
            break


def main(download_dst="resources/data/openml", seed: int = 100) -> None:
    logger.info("Searching for classification tasks...")
    tasks_ = get_tasks()

    logger.info("Filtering out tasks...")
    tasks_ = filter_tasks(tasks_, ADDITIONAL_CONDITIONS)

    logger.info(f"Downloading data to location: {download_dst} ...")
    download_tasks(tasks_, download_dst, seed=seed)


if __name__ == "__main__":
    main()
