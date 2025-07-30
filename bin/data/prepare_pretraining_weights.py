from bin.data.args import get_pretraining_weights_args as get_args
from adaptivee.encoders import DummyEncoder
from adaptivee.ensembler import AdaptiveEnsembler
from adaptivee.target_weights import SoftMaxWeighter, OneHotWeighter
import numpy as np
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin
from pathlib import Path
from config.models import MODELS_LISTS
from loguru import logger


def create_y_target(
    X: NDArray[np.float32],
    y: NDArray[np.float32],
    models: list[ClassifierMixin],
    alpha: float,
    use_onehot: bool,
) -> NDArray[np.float32]:
    if use_onehot:
        target_weighter = OneHotWeighter()
    else:
        target_weighter = SoftMaxWeighter(alpha=alpha)

    ensembler = AdaptiveEnsembler(
        models=models,
        target_weighter=target_weighter,
        encoder=DummyEncoder(),
        is_models_trained=True,
    )
    ensembler.create_adaptive_ensembler(X, y)

    preds = ensembler._get_models_preds(X)
    y_target = target_weighter.get_target_weights(preds, y)

    y_preds = ensembler._get_models_preds(X)
    return y_target, y_preds


def main() -> None:
    args = get_args()

    input_path = Path(args.input_path)
    Models = MODELS_LISTS[args.model_list_id]
    use_onehot = args.use_onehot
    alpha = args.alpha

    for path in input_path.glob("*.npz"):
        logger.info(f"Start {path.stem}")
        models = [Model() for Model in Models]

        data = np.load(path)
        data = dict(data)
        X, y = data["X"], data["y"]

        for model in models:
            model.fit(X, y)

        y_target, y_preds = create_y_target(X, y, models, alpha, use_onehot)

        if args.use_onehot:
            target_name = f"{args.model_list_id}-onehot"
        else:
            target_name = f"{args.model_list_id}-alpha={args.alpha}"

        data[target_name] = y_target
        data[f"{args.model_list_id}-PREDS"] = y_preds
        np.savez(path, **data)


if __name__ == "__main__":
    main()
