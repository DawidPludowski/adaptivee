from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from adaptivee.encoders import DummyEncoder
from adaptivee.ensembler import AdaptiveEnsembler
from adaptivee.target_weights import (
    MixInStaticTargetWeighter,
    OneHotWeighter,
    SoftMaxWeighter,
)
from analysis.configs import MODELS, TARGET_WEIGHTERS


def _get_model() -> list[any]:
    models = [model() for model in MODELS[0]]
    return models


def process_for_weighter(
    target_weighter: MixInStaticTargetWeighter,
    X: np.ndarray,
    y: np.ndarray,
    weighter_name: str,
    dst_path: str,
    data_name: str,
) -> None:
    models = _get_model()
    (Path(dst_path) / f"{weighter_name}").mkdir(exist_ok=True, parents=True)

    for model in models:
        model.fit(X, y)

    ensembler = AdaptiveEnsembler(
        models=models,
        target_weighter=target_weighter,
        encoder=DummyEncoder(),
        is_models_trained=True,
    )
    ensembler.create_adaptive_ensembler(X.to_numpy(), y.to_numpy())

    preds = ensembler._get_models_preds(X)
    y_target = target_weighter.get_target_weights(preds, y.to_numpy())

    df = pd.DataFrame(X)
    df_target = pd.DataFrame(y_target)
    df_target.rename(
        columns={idx: f"target_{idx}" for idx in range(y_target.shape[1])},
        inplace=True,
    )

    df = pd.concat([df, df_target], axis=1)
    df.to_csv(Path(dst_path) / weighter_name / f"{data_name}.csv", index=False)


def main(
    source_path: str = "resources/data/openml/encoder",
    dst_path: str = "resources/data/openml/encoder-mod",
) -> None:

    cnt = 0

    for data_path in Path(source_path).glob("*.csv"):
        df = pd.read_csv(data_path)
        X, y = df.iloc[:, :-1], df.iloc[:, -1]

        for TargetWeighter in TARGET_WEIGHTERS:
            logger.info(
                f"run {type(TargetWeighter()).__name__} on {data_path.stem}; cnt={cnt}"
            )
            process_for_weighter(
                TargetWeighter(),
                X,
                y,
                type(TargetWeighter()).__name__,
                dst_path,
                data_path.stem,
            )

            cnt += 1


if __name__ == "__main__":
    main()
