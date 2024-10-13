from datetime import datetime
from functools import partial
from itertools import product

from loguru import logger

from adaptivee.encoders import DummyEncoder, LiltabEncoder
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
        cls_name = f"{obj.__getattribute__('func').__name__};"
        for key, val in obj.__getattribute__("keywords").items():
            cls_name += f"{key}={val};"
    elif isinstance(obj, type):
        cls_name = obj.__name__
    else:
        cls_name = type(obj).__name__

    if cls_name[-1] == ";":
        cls_name = cls_name[:-1]

    return cls_name


def main() -> None:

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # logger.info("STATIC APPROACHES")

    # for df_train, df_test, df_val, data_name in get_data(
    #     "resources/data/openml"
    # ):

    #     X_train, y_train = (
    #         df_train.iloc[:, :-1].to_numpy(),
    #         df_train.iloc[:, -1].to_numpy(),
    #     )
    #     X_val, y_val = (
    #         df_val.iloc[:, :-1].to_numpy(),
    #         df_val.iloc[:, -1].to_numpy(),
    #     )
    #     X_test, y_test = (
    #         df_test.iloc[:, :-1].to_numpy(),
    #         df_test.iloc[:, -1].to_numpy(),
    #     )

    #     logger.info(f"Start data: {data_name}")

    #     for idx, combination in enumerate(
    #         product(STATIC_TARGET_WEIGHTERS, MODELS)
    #     ):
    #         logger.info(f"Start experiment #{idx:02}")

    #         encoder = DummyEncoder
    #         reweighter = SimpleReweight
    #         target_weighter = combination[0]
    #         models = combination[1]

    #         report = AutoReport(
    #             X_train,
    #             y_train,
    #             X_val,
    #             y_val,
    #             X_test,
    #             y_test,
    #             models,
    #             target_weighter,
    #             encoder,
    #             reweighter,
    #             report_name=f"{timestamp}/{data_name}/{__get_class_name(encoder)}"
    #             f"_{__get_class_name(reweighter)}_{__get_class_name(target_weighter)}",
    #             data_name=data_name,
    #         )

    #         report.make_report()

    logger.info("DYNAMIC APPROACHES")

    for df_train, df_test, df_val, data_name in get_data(
        "resources/data/openml"
    ):

        X_train, y_train = (
            df_train.iloc[:, :-1].to_numpy(),
            df_train.iloc[:, -1].to_numpy(),
        )
        X_val, y_val = (
            df_val.iloc[:, :-1].to_numpy(),
            df_val.iloc[:, -1].to_numpy(),
        )
        X_test, y_test = (
            df_test.iloc[:, :-1].to_numpy(),
            df_test.iloc[:, -1].to_numpy(),
        )

        logger.info(f"Start data: {data_name}")

        for idx, combination in enumerate(
            product(REWEIGHTERS, TARGET_WEIGHTERS, MODELS)
        ):

            target_weighter_name = __get_class_name(combination[1])
            encoder = partial(
                LiltabEncoder,
                model_path=f"resources/models/model_{target_weighter_name}.ckpt",
            )

            logger.info(f"Start experiment #{idx:02}")

            reweighter = combination[0]
            target_weighter = combination[1]
            models = combination[2]

            report = AutoReport(
                X_train,
                y_train,
                X_val,
                y_val,
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
