import pickle as pkl
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from adaptivee.reweighting import DirectionReweight
from adaptivee.target_weights import (
    SoftMaxWeighter,
    StaticFixedWeights,
    StaticGridWeighter,
)
from analysis.ensembler import AdaptiveEnsembler


class AutoReport:

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        Models: list[type] | None = None,
        TargetWeighter: type = None,
        Encoder: type = None,
        Reweighter: type = None,
        meta_data: dict[str, str] = {},
        result_dir: str | Path = "report",
        report_name: str = None,
        data_name: str = None,
        models_from_file: bool = False,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        if Models is not None:
            self.models = [Model() for Model in Models]
        else:
            self.models = None

        if models_from_file:
            models, weights = self.__get_models_weights(data_name)
            self.models = models

        self.target_weighter = TargetWeighter()
        self.encoder = Encoder()
        self.reweighter = Reweighter()
        self.static_weighter = (
            StaticFixedWeights(None)
            if self.models is None
            else StaticGridWeighter()
        )

        self.ensemble = AdaptiveEnsembler(
            self.models,
            self.encoder,
            self.target_weighter,
            self.reweighter,
            static_weighter=self.static_weighter,
            is_models_trained=True,
            use_autogluon=True if self.models is None else False,
        )

        if models_from_file:
            self.ensemble.static_weighter.weights = weights
            self.ensemble.static_weights = weights

        self.root_dir = Path(result_dir) / report_name
        self.meta_data = meta_data
        self.data_name = data_name

    def __get_models_weights(
        self, data_name: str, root_path: str = "resources/models/autogluon"
    ) -> tuple[list[any], np.ndarray]:
        print(data_name)
        model_path = Path(root_path) / data_name / "models.pkl"
        weights_path = Path(root_path) / data_name / "weights.pkl"

        with open(model_path, "rb") as f:
            models = pkl.load(f)
        with open(weights_path, "rb") as f:
            weights = pkl.load(f)
        if len(models) == 1:
            logger.warning("Only one model in file")

        logger.info(f"Number of models: {len(models)}")

        return models, weights

    def make_report(self) -> None:
        self.root_dir.mkdir(exist_ok=True, parents=True)

        self.vizualize_data()
        self.make_experiment()
        self.put_meta_data()

    def vizualize_data(self) -> None:

        y = self.y_train
        if "data_name" in self.meta_data.keys():
            title = self.meta_data["data_name"]
        else:
            title = "Data visualization"

        if self.X_train.shape[1] > 2:
            title += f" (PCA projection from {self.X_train.shape[1]} dims)"

            pca = PCA()
            X = pca.fit_transform(self.X_train)
        else:
            X = self.X_train

        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.title(title)
        plt.savefig(f"{self.root_dir}/data_viz.png")
        plt.clf()

    def make_experiment(self) -> None:

        X_train, y_train, X_val, y_val, X_test, y_test = (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        )
        self.ensemble.create_adaptive_ensembler(X_train, y_train, X_val, y_val)

        self.__report_metrics(X_train, y_train, X_test, y_test)
        self.__report_weights(X_train, y_train, X_test, y_test)
        self.__save_models()

    def __save_models(self) -> None:
        models = self.ensemble.models
        weights = self.ensemble.static_weights

        with open(self.root_dir / "models.pkl", "wb") as f:
            pkl.dump(models, f)

        with open(self.root_dir / "weights.pkl", "wb") as f:
            pkl.dump(weights, f)

    def put_meta_data(self) -> None:
        meta = {
            "data": {
                "data_name": self.data_name,
                "train_size": self.X_train.shape[0],
                "val_size": self.X_val.shape[0],
                "test_size": self.X_test.shape[0],
                "train_y_ratio": float((self.y_train == 1).mean()),
                "n_features": self.X_train.shape[1],
            },
            "reweighter": {
                "name": type(self.reweighter).__name__,
            },
            "target_weighter": {
                "name": type(self.target_weighter).__name__,
            },
            "encoder": type(self.encoder).__name__,
            "meta_data": self.meta_data,
            "models": {
                "n": len(self.ensemble.models),
                "weights": self.ensemble.static_weights.tolist(),
                "names": [
                    type(model_name).__name__
                    for model_name in self.ensemble.models
                ],
            },
        }

        if isinstance(self.reweighter, DirectionReweight):
            meta["reweighter"]["step_size"] = float(self.reweighter.step_size)

            X_test, y_test = self.X_test, self.y_test
            y_pred = self.ensemble._get_models_preds(X_test)
            static_weights = self.ensemble.get_weights_static(X_test)
            encoder_weights = self.encoder.predict(X_test)
            self.reweighter.optimize_hp(
                y_true=y_test,
                y_pred=y_pred,
                static_weights=static_weights,
                encoder_weights=encoder_weights,
            )
            meta["reweighter"]["step_size_test_optim"] = float(
                self.reweighter.step_size
            )

        if isinstance(self.target_weighter, SoftMaxWeighter):
            meta["target_weighter"]["alpha"] = float(
                self.target_weighter.alpha
            )

        with open(self.root_dir / "meta_data.yaml", "w") as f:
            yaml.dump(meta, f)

    # def __get_train_test_split(
    #     self, random_seed: int = 123, test_size: float = 0.3
    # ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    #     X_train, X_test, y_train, y_test = train_test_split(
    #         self.X, self.y, test_size=test_size, random_state=random_seed
    #     )
    #     return X_train, y_train, X_test, y_test

    def __report_metrics(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:

        y_train_pred = self.ensemble.predict(X_train)
        y_test_pred = self.ensemble.predict(X_test)

        y_train_pred_bin = (y_train_pred > 0.5).astype(int)
        y_test_pred_bin = (y_test_pred > 0.5).astype(int)

        metrics_dynamic = {
            0: ["acc", "train", accuracy_score(y_train, y_train_pred_bin)],
            1: ["acc", "test", accuracy_score(y_test, y_test_pred_bin)],
            2: ["roc-auc", "train", roc_auc_score(y_train, y_train_pred)],
            3: ["roc-auc", "test", roc_auc_score(y_test, y_test_pred)],
            4: [
                "acc-b",
                "train",
                balanced_accuracy_score(y_train, y_train_pred_bin),
            ],
            5: [
                "acc-b",
                "test",
                balanced_accuracy_score(y_test, y_test_pred_bin),
            ],
            6: ["f1", "train", f1_score(y_train, y_train_pred_bin)],
            7: ["f1", "test", f1_score(y_test, y_test_pred_bin)],
        }

        df_dynamic = pd.DataFrame(data=metrics_dynamic).T
        df_dynamic.columns = ["metric", "data", "score"]
        df_dynamic["type"] = "dynamic"

        y_train_pred = self.ensemble.predict_static(X_train)
        y_test_pred = self.ensemble.predict_static(X_test)

        y_train_pred_bin = (y_train_pred > 0.5).astype(int)
        y_test_pred_bin = (y_test_pred > 0.5).astype(int)

        metrics_static = {
            0: ["acc", "train", accuracy_score(y_train, y_train_pred_bin)],
            1: ["acc", "test", accuracy_score(y_test, y_test_pred_bin)],
            2: ["roc-auc", "train", roc_auc_score(y_train, y_train_pred)],
            3: ["roc-auc", "test", roc_auc_score(y_test, y_test_pred)],
            4: [
                "acc-b",
                "train",
                balanced_accuracy_score(y_train, y_train_pred_bin),
            ],
            5: [
                "acc-b",
                "test",
                balanced_accuracy_score(y_test, y_test_pred_bin),
            ],
            6: ["f1", "train", f1_score(y_train, y_train_pred_bin)],
            7: ["f1", "test", f1_score(y_test, y_test_pred_bin)],
        }

        df_static = pd.DataFrame(data=metrics_static).T
        df_static.columns = ["metric", "data", "score"]
        df_static["type"] = "static"

        metric_bounds = {
            1: [
                "acc",
                "test",
                self.__get_metric_bounds(accuracy_score, is_prob=False),
            ],
            3: [
                "roc-auc",
                "test",
                self.__get_metric_bounds(roc_auc_score, is_prob=True),
            ],
            5: [
                "acc-b",
                "test",
                self.__get_metric_bounds(
                    balanced_accuracy_score, is_prob=False
                ),
            ],
            7: [
                "f1",
                "test",
                self.__get_metric_bounds(f1_score, is_prob=False),
            ],
        }
        df_bounds = pd.DataFrame(data=metric_bounds).T
        df_bounds.columns = ["metric", "data", "score"]
        df_bounds["type"] = "upper_bound"

        pd.concat(
            [df_dynamic, df_static, df_bounds], ignore_index=True, axis=0
        ).to_csv(self.root_dir / "metrics.csv", index=False)

    def __report_weights(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        train_weights = self.ensemble.get_weights(X_train)
        test_weights = self.ensemble.get_weights(X_test)

        y_train_preds_all = self.ensemble._get_models_preds(X_train)
        y_test_preds_all = self.ensemble._get_models_preds(X_test)
        target_train_weights = self.target_weighter.get_target_weights(
            y_train_preds_all, y_train
        )
        target_test_weights = self.target_weighter.get_target_weights(
            y_test_preds_all, y_test
        )

        static_train_weights = self.static_weighter.get_target_weights(
            y_train_preds_all, y_train
        )
        static_test_weights = self.static_weighter.get_target_weights(
            y_test_preds_all, y_test
        )

        # csv

        train_weights_df = pd.DataFrame(data=train_weights)
        train_weights_df["type"] = "train_weights"
        train_weights_df.reset_index(drop=False, inplace=True)

        test_weights_df = pd.DataFrame(data=test_weights)
        test_weights_df["type"] = "test_weights"
        test_weights_df.reset_index(drop=False, inplace=True)

        target_train_weights_df = pd.DataFrame(data=target_train_weights)
        target_train_weights_df["type"] = "target_train_weights"
        target_train_weights_df.reset_index(drop=False, inplace=True)

        target_test_weights_df = pd.DataFrame(data=target_test_weights)
        target_test_weights_df["type"] = "target_test_weights"
        target_test_weights_df.reset_index(drop=False, inplace=True)

        static_train_weights_df = pd.DataFrame(data=static_train_weights)
        static_train_weights_df["type"] = "static_train_weights"
        static_train_weights_df.reset_index(drop=False, inplace=True)

        static_test_weights_df = pd.DataFrame(data=static_test_weights)
        static_test_weights_df["type"] = "static_test_weights"
        static_test_weights_df.reset_index(drop=False, inplace=True)

        df = pd.concat(
            [
                train_weights_df,
                test_weights_df,
                target_train_weights_df,
                target_test_weights_df,
                static_train_weights_df,
                static_test_weights_df,
            ]
        )

        df.to_csv(self.root_dir / "weights.csv", index=False)

        # viz

        train_differences = np.linalg.norm(
            train_weights - target_train_weights, axis=1
        )
        test_differences = np.linalg.norm(
            test_weights - target_test_weights, axis=1
        )

        temp_train_df = pd.DataFrame()
        temp_train_df["diff"] = train_differences
        temp_train_df["type"] = "train"

        temp_test_df = pd.DataFrame()
        temp_test_df["diff"] = test_differences
        temp_test_df["type"] = "test"

        g = sns.boxplot(
            data=pd.concat([temp_train_df, temp_test_df]),
            x="type",
            y="diff",
        )
        g.set(title="Difference in norm of predicted and target weights")
        plt.ylim((0, 1))

        plt.savefig(f"{self.root_dir}/weights_diff.png")
        plt.clf()

    def __get_metric_bounds(
        self, metric: callable, is_prob: bool = False
    ) -> float:

        X_test, y_test = self.X_test, self.y_test

        if isinstance(y_test, pd.Series):
            y_test = y_test.to_numpy()

        y_test = y_test.reshape(-1, 1)

        # dynamic
        y_pred = self.ensemble._get_models_preds(X_test)
        diff = np.abs(y_test - y_pred)

        best_pred_idx = np.argmin(diff, axis=1)
        best_pred = y_pred[np.arange(y_pred.shape[0]), best_pred_idx]

        if not is_prob:
            best_pred = (0.5 < best_pred).astype(int)

        metric_best = metric(y_test, best_pred)

        return metric_best


class AutoSummaryReport:

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def make_report(self) -> None:
        self.compare_metrics()
        self.compare_weights()

    def compare_metrics(self) -> None:

        metrics = self.__get_metrics()
        self.__make_rank_test(metrics)
        self.__summarize_metrics(metrics)

    def compare_weights(self) -> None:
        warnings.warn("Compare weights not implemented")
        weights = self.__get_weights()
        self.__compare_weights_diversity(weights)
        self.__compare_weights_predictions(weights)

    def __get_metrics(self) -> pd.DataFrame:
        metrics = pd.DataFrame()

        metrics_files = self.root_dir.rglob("metrics.csv")
        for metric_file in metrics_files:
            data_name = metric_file.parent.parent.stem
            experiment_name = metric_file.parent.stem

            metric_summary = pd.read_csv(metric_file)
            metric_summary["data_name"] = data_name
            metric_summary["experiment_name"] = experiment_name

            metrics = pd.concat([metrics, metric_summary])

        return metrics

    def get_metrics(self):
        return self.__get_metrics()

    def get_weights(self):
        return self.__get_weights()

    def get_weights_metrics(self):
        summary_dfs = []

        dirs = self.root_dir.glob("*")
        for dir_ in dirs:
            weights_files = dir_.rglob("weights.csv")
            for weight_file in weights_files:
                data_name = weight_file.parent.parent.parent.parent.stem
                experiment_name = weight_file.parent.parent.parent.stem

                weights_summary = pd.read_csv(weight_file)

                test_weights = (
                    weights_summary[weights_summary["type"] == "test_weights"]
                    .iloc[:, 1:-1]
                    .to_numpy()
                )
                target_test_weights = (
                    weights_summary[
                        weights_summary["type"] == "target_test_weights"
                    ]
                    .iloc[:, 1:-1]
                    .to_numpy()
                )

                diffs = test_weights - target_test_weights
                norm_l2 = np.linalg.norm(diffs)
                std_res = np.std(test_weights, axis=0).mean()
                std_target = np.std(target_test_weights, axis=0).mean()

                summary_df = pd.DataFrame(
                    data={
                        "data_name": [data_name],
                        "experiment_name": [experiment_name],
                        "norm": [norm_l2],
                        "std": [std_res],
                        "std_target": [std_target],
                    }
                )
                summary_dfs.append(summary_df)

        summary = pd.concat(summary_dfs)
        return summary

    def __make_rank_test(self, metrics: pd.DataFrame) -> None:
        warnings.warn("Rank test not implemented")

    def __summarize_metrics(self, metrics: pd.DataFrame) -> None:

        for metric_name in metrics["metric"].unique():

            g = sns.boxplot(
                data=metrics[metrics["metric"] == metric_name],
                x="experiment_name",
                y="score",
                hue="data",
            )
            g.set(ylabel=f"{metric_name}", title=f"{metric_name} score")
            g.set_ylim(-0.1, 1.1)
            # plt.figure(figsize=(16, 9))
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(f"{self.root_dir}/score_{metric_name}")
            plt.clf()

    def __get_weights(self) -> pd.DataFrame:

        weights_dfs = []

        dirs = self.root_dir.glob("*")
        for dir_ in dirs:
            weights_files = dir_.rglob("weights.csv")
            for weight_file in weights_files:
                data_name = weight_file.parent.parent.stem
                experiment_name = weight_file.parent.stem

                weights_summary = pd.read_csv(weight_file)
                weights_summary["data_name"] = data_name
                weights_summary["experiment_name"] = experiment_name

                weights_dfs.append(weights_summary)

        weights = pd.concat(weights_dfs)
        return weights

    def __compare_weights_diversity(self, weights: pd.DataFrame) -> None:
        pass

    def __compare_weights_predictions(self, weights: pd.DataFrame) -> None:
        pass
