import logging
from datetime import datetime
from functools import partial

from loguru import logger

from adaptivee.encoders import LiltabEncoder
from adaptivee.reweighting import DirectionReweight
from adaptivee.target_weights import OneHotWeighter, SoftMaxWeighter
from analysis.auto_report import AutoReport, AutoSummaryReport
from analysis.data.openml import get_data

logging.getLogger().disabled = True


def __get_class_name(obj: any) -> str:
    if isinstance(obj, partial):
        cls_name = f"{obj.__getattribute__('func').__name__}"
    elif isinstance(obj, type):
        cls_name = obj.__name__
    else:
        cls_name = type(obj).__name__

    return cls_name


def main() -> None:

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for df_train, df_val, df_test, data_name in get_data("resources/data/openml"):

        X_train, y_train = (
            df_train.iloc[:, :-1],
            df_train.iloc[:, -1],
        )
        X_val, y_val = (
            df_val.iloc[:, :-1],
            df_val.iloc[:, -1]
        )
        X_test, y_test = (
            df_test.iloc[:, :-1],
            df_test.iloc[:, -1],
        )

        logger.info(f"Start data: {data_name}")

        encoder = partial(
            LiltabEncoder, model_path="resources/models/full_model.ckpt"
        )
        reweighter = partial(DirectionReweight, step_size=0.1)
        target_weighter = SoftMaxWeighter

        report = AutoReport(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            Models=None,
            TargetWeighter=target_weighter,
            Encoder=encoder,
            Reweighter=reweighter,
            report_name=f"{timestamp}/{data_name}/{__get_class_name(encoder)}"
            f"_{__get_class_name(reweighter)}_{__get_class_name(target_weighter)}",
            data_name=data_name,
        )

        report.make_report()

    auto_summary_report = AutoSummaryReport(f"report/{timestamp}")
    auto_summary_report.make_report()


if __name__ == "__main__":
    main()
