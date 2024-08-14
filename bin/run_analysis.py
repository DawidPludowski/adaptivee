from datetime import datetime
from functools import partial
from itertools import product

from loguru import logger

from adaptivee.encoders import DummyEncoder
from adaptivee.reweighting import SimpleReweight
from analysis.auto_report import AutoReport, AutoSummaryReport
from analysis.configs import REWEIGHTERS  # DATASETS,
from analysis.configs import (
    ENCODERS,
    MODELS,
    STATIC_TARGET_WEIGHTERS,
    TARGET_WEIGHTERS,
)
from analysis.data.openml import get_data


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

    logger.info("STATIC APPROACHES")

    for df_train, df_test, data_name in get_data("resources/data/openml"):

        X_train, y_train = (
            df_train.iloc[:, :-1].to_numpy(),
            df_train.iloc[:, -1].to_numpy(),
        )
        X_test, y_test = (
            df_test.iloc[:, :-1].to_numpy(),
            df_test.iloc[:, -1].to_numpy(),
        )

        logger.info(f"Start data: {data_name}")

        for idx, combination in enumerate(
            product(STATIC_TARGET_WEIGHTERS, MODELS)
        ):
            logger.info(f"Start experiment #{idx:02}")

            encoder = DummyEncoder
            reweighter = SimpleReweight
            target_weighter = combination[0]
            models = combination[1]

            report = AutoReport(
                X_train,
                y_train,
                X_test,
                y_test,
                models,
                target_weighter,
                encoder,
                reweighter,
                report_name=f"{timestamp}/{data_name}/{__get_class_name(encoder)}"
                f"_{__get_class_name(reweighter)}_{__get_class_name(target_weighter)}",
                data_name=data_name,
            )

            report.make_report()

    logger.info("DYNAMIC APPROACHES")

    for df_train, df_test, data_name in get_data("resources/data/openml"):

        X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
        X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]

        logger.info(f"Start data: {data_name}")

        for idx, combination in enumerate(
            product(ENCODERS, REWEIGHTERS, TARGET_WEIGHTERS, MODELS)
        ):

            logger.info(f"Start experiment #{idx:02}")

            encoder = combination[0]
            reweighter = combination[1]
            target_weighter = combination[2]
            models = combination[3]

            report = AutoReport(
                X_train,
                y_train,
                X_test,
                y_test,
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
