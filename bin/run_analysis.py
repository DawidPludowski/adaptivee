from datetime import datetime
from functools import partial
from itertools import product

from loguru import logger

from analysis.auto_report import AutoReport, AutoSummaryReport
from analysis.configs import (
    DATASETS,
    ENCODERS,
    MODELS,
    REWEIGHTERS,
    TARGET_WEIGHTERS,
)


def __get_class_name(obj: any) -> str:
    if isinstance(obj, partial):
        cls_name = obj.__getattribute__("func").__name__
    elif isinstance(obj, type):
        cls_name = obj.__name__
    else:
        cls_name = type(obj).__name__

    return cls_name


def main() -> None:

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for idx, combination in enumerate(
        product(DATASETS, ENCODERS, REWEIGHTERS, TARGET_WEIGHTERS, MODELS)
    ):

        logger.info(f"Start experiment #{idx:02}")

        data_name = combination[0][0]
        X, y = combination[0][1][0], combination[0][1][1]
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
            report_name=f"{timestamp}/{data_name}/{__get_class_name(encoder)}"
            f"_{__get_class_name(reweighter)}_{__get_class_name(target_weighter)}",
            data_name=data_name,
        )

        report.make_report()

    auto_summary_report = AutoSummaryReport(f"report/{timestamp}")
    auto_summary_report.make_report()


if __name__ == "__main__":
    main()
