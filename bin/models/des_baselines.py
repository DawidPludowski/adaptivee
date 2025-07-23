from numpy.typing import NDArray
import numpy as np
from typing import Any
from pathlib import Path
from bin.models.args import get_des_baselines_args as get_args
from bin.utils import get_trained_models, get_metrics
from config.models import BASELINES_LIST
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import json
from sklearn.ensemble import StackingClassifier
from functools import partial
from loguru import logger


def main() -> None:
    args = get_args()

    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    model_list_id = args.model_list_id
    out_path = args.out_path

    if out_path is None:
        out_path = test_path.parent / "results.json"

    train_data = dict(np.load(train_path))
    test_data = dict(np.load(test_path))

    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    metrics = {}
    for baseline_name, BASELINE_CLS in BASELINES_LIST.items():
        logger.info(f"Start {baseline_name}, {train_path.name}")
        models = get_trained_models(model_list_id, X_train, y_train)
        if (
            isinstance(BASELINE_CLS, partial)
            and BASELINE_CLS.func is StackingClassifier
        ):
            models_ = [(f"{idx}", model) for idx, model in enumerate(models)]
            baseline = BASELINE_CLS(estimators=models_)
        else:
            baseline = BASELINE_CLS(pool_classifiers=models)

        baseline.fit(X_train, y_train)
        y_train_p = baseline.predict(X_train)
        y_test_p = baseline.predict(X_test)
        metrics[baseline_name] = get_metrics(
            y_train, y_train_p, y_test, y_test_p
        )

        if args.save_out:
            train_data[f"{baseline_name}_pred"] = y_train_p
            test_data[f"{baseline_name}_pred"] = y_test_p

    with open(out_path, "w") as f:
        json.dump(metrics, f)

    if args.save_out:
        np.savez(train_path, **dict(train_data))
        np.savez(test_path, **dict(test_data))


if __name__ == "__main__":
    main()
