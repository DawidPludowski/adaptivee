from datetime import datetime
from itertools import product

from loguru import logger

from analysis.auto_report import AutoReport
from analysis.configs import (
    DATASETS,
    ENCODERS,
    MODELS,
    REWEIGHTERS,
    TARGET_WEIGHTERS,
)


def main() -> None:

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for idx, combination in enumerate(
        product(DATASETS, ENCODERS, REWEIGHTERS, TARGET_WEIGHTERS, MODELS)
    ):

        logger.info(f"Start experiment #{idx:02}")

        X, y = combination[0][0], combination[0][1]
        encoder = combination[1]
        reweighter = combination[2]
        target_weighter = combination[3]
        models = combination[4]

        report = AutoReport(
            X,
            y,
            models,
            target_weighter,
            encoder,
            reweighter,
            report_name=f"{timestamp}/experiment_{idx:02}",
        )

        report.make_report()


if __name__ == "__main__":
    main()
