import warnings
from typing import Literal

import numpy as np
import pandas as pd
from autogluon.core.models.ensemble.weighted_ensemble_model import (
    WeightedEnsembleModel,
)
from autogluon.tabular import TabularPredictor

from adaptivee.encoders import MixInEncoder
from adaptivee.reweighting import MixInReweight, SimpleReweight
from adaptivee.target_weights import (
    MixInStaticTargetWeighter,
    MixInTargetWeighter,
    SoftMaxWeighter,
    StaticFixedWeights,
    StaticGridWeighter,
)


class AdaptiveEnsembler:

    def __init__(
        self,
        models: list[any] | None,
        encoder: MixInEncoder,
        target_weighter: MixInTargetWeighter = SoftMaxWeighter(),
        reweighter: MixInReweight = SimpleReweight(),
        static_weighter: MixInStaticTargetWeighter = StaticGridWeighter(),
        is_models_trained: bool = True,
        predict_fn: str = "predict_proba",
        train_fn: str = "fit",
        use_autogluon: bool = False,
        autogluon_fit_kwargs: dict[str, any] = {
            "num_stack_levels": 0,
            "num_bag_sets": 2,
            "num_bag_folds": 5,
            "verbosity": 0,
            # "presets": "best_quality",
            "time_limit": 5 * 60,
        },
    ) -> None:
        self.models = models
        self.encoder = encoder
        self.target_weighter = target_weighter
        self.reweighter = reweighter

        if use_autogluon and not isinstance(
            target_weighter, StaticFixedWeights
        ):
            if not isinstance(target_weighter, MixInStaticTargetWeighter):
                warnings.warn(
                    "autogluon models used but dynamic target weighter used. Autogluon's weights will be overrided."
                )
            else:
                warnings.warn(
                    f"autogluon models used but StaticFixedWeights not used. Autogluon's "
                    f"weights will override {type(target_weighter).__name__} weights."
                )
        if use_autogluon and models is not None:
            warnings.warn(
                "autogluon models used but custom models list provided. `models` argument will be ignored."
            )

        if isinstance(target_weighter, MixInStaticTargetWeighter):
            warnings.warn(
                f"Change static_weighter ({type(static_weighter).__name__})"
                f"to the target_weighter ({type(target_weighter).__name__})"
            )
            self.static_weighter = target_weighter
        else:
            self.static_weighter = static_weighter

        if not use_autogluon and models is None:
            warnings.warn(
                "models were not provided but `use_autogluon` is set to False."
            )

        self.predict_fn = predict_fn
        self.train_fn = train_fn

        self.is_models_trained = is_models_trained

        self.static_weights = None
        self.autogluon_fit_kwargs = autogluon_fit_kwargs
        self.use_autogluon = use_autogluon

    def create_adaptive_ensembler(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        return_score: bool = False,
    ) -> None:

        if self.use_autogluon:
            self._train_autogluon_ensemble(X, y)

        if not self.is_models_trained:
            self._train_models(X, y)

        y_pred = self._get_models_preds(X)
        weights = self.target_weighter.get_target_weights(y_pred, y)
        static_weights = self.static_weighter._find_best_weights(y_pred, y)

        self.static_weights = static_weights

        if not isinstance(self.target_weighter, MixInStaticTargetWeighter):
            self.encoder.train(X, weights)

            if X_val is not None and y_val is not None:
                self.tune_reweighter(X_val, y_val)

        if return_score:
            raise NotImplementedError()

    def tune_reweighter(self, X: np.ndarray, y: np.ndarray) -> None:
        encoder_weights = self.encoder.predict(X)
        y_pred = self._get_models_preds(X)
        self.reweighter.optimize_hp(
            y_true=y,
            y_pred=y_pred,
            static_weights=self.static_weights,
            encoder_weights=encoder_weights,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_preds = self._get_models_preds(X)

        final_weights = self.get_weights(X)

        y_pred_final = np.sum(y_preds * final_weights, axis=1)
        return y_pred_final

    def predict_static(self, X: np.ndarray) -> np.ndarray:
        y_preds = self._get_models_preds(X)
        final_weights = self.get_weights_static(X)

        y_pred_final = np.sum(y_preds * final_weights, axis=1)
        return y_pred_final

    def get_weights_static(self, X: np.ndarray) -> np.ndarray:
        reweights = self.static_weights.reshape(1, -1).repeat(
            repeats=X.shape[0], axis=0
        )
        return reweights

    def get_weights(self, X: np.ndarray) -> np.ndarray:
        if isinstance(self.target_weighter, MixInStaticTargetWeighter):
            reweights = self.static_weights.reshape(1, -1).repeat(
                repeats=X.shape[0], axis=0
            )

        else:
            weights = self.encoder.predict(X)
            reweights = self.reweighter.get_final_weights(
                weights, self.static_weights
            )

        return reweights

    def _train_models(self, X: np.ndarray, y: np.ndarray) -> None:

        if self.train_fn != "fit":
            raise NotImplementedError(
                'Using function other than "train" for training is not supported'
            )

        for model in self.models:
            model.fit(X, y)

        self.is_models_trained = True

    def _get_models_preds(self, X: np.ndarray) -> np.ndarray:

        if not self.is_models_trained:
            raise Exception("Cannot get predicitons from not trained models.")

        if self.predict_fn != "predict_proba":
            raise NotImplementedError(
                'Using function other than "predict_proba" for prediction is not supported'
            )

        y_preds = []
        for model in self.models:
            y_pred = model.predict_proba(X)
            if not self.use_autogluon:
                y_pred = y_pred[:, 1]
            y_preds.append(y_pred.reshape(-1, 1))

        y_preds = np.hstack(y_preds)

        return y_preds

    def _train_autogluon_ensemble(self, X: np.ndarray, y: np.ndarray) -> None:
        models, weights = self.__get_autogluon_ensemble(X, y)

        weights = np.array(weights)
        self.static_weighter.weights = weights

        self.models = models
        self.is_models_trained = True

    def __get_autogluon_ensemble(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[list[any], np.ndarray]:
        df = pd.DataFrame(X)
        df["class"] = y

        predictor = TabularPredictor(label="class")

        out_path = predictor.path
        warnings.warn(
            f"The AutoGluon output will be stored in {out_path} directory."
        )

        predictor.fit(train_data=df, **self.autogluon_fit_kwargs)

        models_names = predictor.leaderboard()["model"]

        if models_names.shape[0] == 1:
            warnings.warn(
                f"Autogluon ensemble consists of only one model - {models_names[0]}."
            )

        for idx, model_name in models_names.items():
            model = predictor._trainer.load_model(model_name)
            if isinstance(model, WeightedEnsembleModel):
                if idx != 0:
                    warnings.warn(
                        f"Autogluon's ensemble is {idx+1}. best model from autogluon, not first."
                    )
                if len(model.models) != 1:
                    warnings.warn(
                        "Autogluon returned multilayer stacking model; only first layer will be used by adaptivee."
                    )

                first_layer = model.models[0]
                weights = first_layer.weights_

                models = []
                for base_model_name in model.base_model_names:
                    base_model = predictor._trainer.load_model(base_model_name)
                    models.append(base_model)

                assert len(weights) == len(models)
                return models, weights
