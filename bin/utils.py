from numpy.typing import NDArray
import numpy as np
from sklearn.base import BaseEstimator
from config.models import MODELS_LISTS

from typing import Any
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_trained_models(
    model_list_id: str,
    X_train: NDArray[np.float32],
    y_train: NDArray[np.float32],
) -> list[BaseEstimator]:
    Models = MODELS_LISTS[model_list_id]
    models = [Model() for Model in Models]
    for model in models:
        model.fit(X_train, y_train)
    return models


def get_metrics(
    y_train_t: NDArray[np.float32],
    y_train_p: NDArray[np.float32],
    y_test_t: NDArray[np.float32],
    y_test_p: NDArray[np.float32],
) -> dict[str, Any]:
    return {
        "train": {
            "accuracy": accuracy_score(y_train_t, y_train_p),
            "roc-auc": roc_auc_score(y_train_t, y_train_p),
            "f1": f1_score(y_train_t, y_train_p),
        },
        "test": {
            "accuracy": accuracy_score(y_test_t, y_test_p),
            "roc-auc": roc_auc_score(y_test_t, y_test_p),
            "f1": f1_score(y_test_t, y_test_p),
        },
    }


def get_generic_preprocessing() -> Pipeline:
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
