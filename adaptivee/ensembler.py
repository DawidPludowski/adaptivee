import numpy as np

from adaptivee.encoders import MixInEncoder
from adaptivee.reweighting import MixInReweight, SimpleReweight
from adaptivee.target_weights import (
    MixInStaticTargetWeighter,
    MixInTargetWeighter,
    SoftMaxWeighter,
    StaticLogisticWeighter,
)


class AdaptiveEnsembler:

    def __init__(
        self,
        models: list[any],
        encoder: MixInEncoder,
        target_weighter: MixInTargetWeighter = SoftMaxWeighter(),
        reweighter: MixInReweight = SimpleReweight(),
        static_weighter: MixInStaticTargetWeighter = StaticLogisticWeighter(),
        is_models_trained: bool = True,
        predict_fn: str = "predict_proba",
        train_fn: str = "fit",
    ) -> None:
        self.models = models
        self.encoder = encoder
        self.target_weighter = target_weighter
        self.reweighter = reweighter
        self.static_weighter = static_weighter

        self.predict_fn = predict_fn
        self.train_fn = train_fn

        self.is_models_trained = is_models_trained

        self.static_weights = None

    def create_adaptive_ensembler(
        self, X: np.ndarray, y: np.ndarray, return_score: bool = False
    ) -> None:

        if not self.is_models_trained:
            self._train_models(X, y)

        y_pred = self._get_models_preds(X)
        weights = self.target_weighter.get_target_weights(y_pred, y)
        static_weights = self.static_weighter._find_best_weights(y_pred, y)

        self.static_weights = static_weights

        if not isinstance(self.target_weighter, MixInStaticTargetWeighter):
            self.encoder.train(X, weights)

        if return_score:
            raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_preds = self._get_models_preds(X)

        final_weights = self.get_weights(X)

        y_pred_final = np.mean(y_preds * final_weights, axis=1)
        return y_pred_final

    def get_weights(self, X: np.ndarray) -> np.ndarray:
        if isinstance(self.target_weighter, MixInStaticTargetWeighter):
            reweights = self.static_weights
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
            y_pred = model.predict_proba(X)[:, 1]
            y_preds.append(y_pred.reshape(-1, 1))

        y_preds = np.hstack(y_preds)

        return y_preds
