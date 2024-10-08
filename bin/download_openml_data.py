import random
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from openml import datasets, study, tasks
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

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


def get_tasks() -> list[int]:
    tasks_ids = study.get_suite(99).tasks
    return tasks_ids


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


def create_split(
    data: pd.DataFrame, target_feature: str, seed: int, train_ratio: float = 0.6, val_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    colnames = data.columns

    df_train, df_test = train_test_split(
        data, stratify=data[target_feature], shuffle=True, random_state=seed,
        train_size=(train_ratio + val_ratio)
    )
    df_train = pd.DataFrame(data=df_train, columns=colnames)
    df_test = pd.DataFrame(data=df_test, columns=colnames)
    
    df_train, df_val = train_test_split(
        df_train, stratify=df_train[target_feature], shuffle=True, random_state=seed,
        train_size=(train_ratio / (train_ratio + val_ratio))
    )
    
    df_train = pd.DataFrame(data=df_train, columns=colnames)
    df_val = pd.DataFrame(data=df_val, columns=colnames)

    return df_train, df_val, df_test


def download_tasks(
    tasks_: list[int],
    download_dst: str | Path,
    max_downloads: int = 100,
    shuffle: bool = True,
    seed: int = 42,
) -> None:

    download_dst = Path(download_dst)
    download_dst.mkdir(exist_ok=True, parents=True)

    (download_dst / "train").mkdir(exist_ok=True, parents=True)
    (download_dst / "val").mkdir(exist_ok=True, parents=True)
    (download_dst / "test").mkdir(exist_ok=True, parents=True)
    (download_dst / "encoder").mkdir(exist_ok=True, parents=True)

    if shuffle:
        random.seed(seed)
        random.shuffle(tasks_)

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

        n_classes = np.unique(data[target_name]).shape[0]
        if n_classes != 2:
            logger.warning(f"Dataset skipped due to class number: {n_classes}")
            continue

        logger.info(f"data size: {data_size} MB")

        data = _ensure_last_target(data, target_name)

        if downloaded_cnt < 20:

            df_train, df_val, df_test = create_split(
                data, target_feature=target_name, seed=seed
            )
            
            pipeline = _get_generic_preprocessing()
            encoder = LabelEncoder()
            
            X_train = pipeline.fit_transform(df_train.iloc[:,:-1])
            y_train = encoder.fit_transform(df_train.iloc[:,-1])
            
            df_train = pd.DataFrame(data=X_train)
            df_train['target'] = y_train
            
            X_val = pipeline.transform(df_val.iloc[:,:-1])
            y_val = encoder.transform(df_val.iloc[:,-1])
            
            df_val = pd.DataFrame(data=X_val)
            df_val['target'] = y_val
            
            X_test = pipeline.transform(df_test.iloc[:,:-1])
            y_test = encoder.transform(df_test.iloc[:,-1])
            
            df_test = pd.DataFrame(data=X_test)
            df_test['target'] = y_test

            df_train.to_csv(
                download_dst / "train" / f"{task_name}.csv", index=False
            )
            df_val.to_csv(
                download_dst / 'val' / f"{task_name}.csv", index=False
            )
            df_test.to_csv(
                download_dst / "test" / f"{task_name}.csv", index=False
            )

        else:
            
            pipeline = _get_generic_preprocessing()
            encoder = LabelEncoder()
            
            X = pipeline.fit_transform(data.iloc[:,:-1])
            y = encoder.fit_transform(data.iloc[:,-1])

            df = pd.DataFrame(data=X)
            df['target'] = y

            df.to_csv(
                download_dst / "encoder" / f"{task_name}.csv", index=False
            )

        downloaded_cnt += 1

        if downloaded_cnt == max_downloads:
            logger.info(
                f"Stopped due to max_donwloads parameter={max_downloads}"
            )
            break

    logger.info(f"Done. Total number of tasks: {downloaded_cnt}")


def main(download_dst="resources/data/openml", seed: int = 100) -> None:

    logger.info("Searching for CC18 AutoML tasks...")
    tasks_ = get_tasks()

    logger.info(f"Downloading data to location: {download_dst} ...")
    download_tasks(tasks_, download_dst, seed=seed)


if __name__ == "__main__":
    main()
