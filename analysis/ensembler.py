import numpy as np
from loguru import logger

from adaptivee.ensembler import AdaptiveEnsembler as Adaptivee
from adaptivee.target_weights import MixInStaticTargetWeighter


class AdaptiveEnsembler(Adaptivee):

    def create_adaptive_ensembler(
        self, X: np.ndarray, y: np.ndarray, return_score: bool = False
    ) -> None:

        if self.models is None:
            logger.info("Create models with AutoGluon...")
            self._train_autogluon_ensemble(X, y)
            n_models = len(self.models)
            if n_models < 2:
                logger.warning(f"Number of models: {n_models}")
            else:
                logger.info(f"Number of models: {n_models}")

        if not self.is_models_trained:
            logger.info("Model training...")
            self._train_models(X, y)

        y_pred = self._get_models_preds(X)
        logger.info("Computing target weights...")
        weights = self.target_weighter.get_target_weights(y_pred, y)

        logger.info("Computing static target weights...")
        static_weights = self.static_weighter._find_best_weights(y_pred, y)

        self.static_weights = static_weights

        if not isinstance(self.target_weighter, MixInStaticTargetWeighter):
            logger.info("Encoder training...")
            self.encoder.train(X, weights)

        if return_score:
            raise NotImplementedError()
