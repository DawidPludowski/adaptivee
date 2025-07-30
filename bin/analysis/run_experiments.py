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
from bin.analysis.args import get_run_experiment_args as get_args
from bin.analysis.utils import (
    load_encoder,
    get_reweighter,
    get_target_weighter,
)
from config.models import MODELS_LISTS
import traceback


def __get_class_name(obj: any) -> str:
    if isinstance(obj, partial):
        cls_name = f"{obj.__getattribute__('func').__name__}"
    elif isinstance(obj, type):
        cls_name = obj.__name__
    else:
        cls_name = type(obj).__name__

    return cls_name


def main() -> None:

    args = get_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for train_data, test_data, data_name in get_data(args.data_dir):

        X_train, y_train = train_data["X"], train_data["y"]
        X_test, y_test = test_data["X"], test_data["y"]

        logger.info(f"Start data: {data_name}")

        encoder = load_encoder(args.encoder_path)
        reweighter = get_reweighter(args.reweighter)
        target_weighter = get_target_weighter(args.alpha)

        if args.use_autogluon:
            models = None
        else:
            models = MODELS_LISTS[args.model_list_id]

        try:

            report = AutoReport(
                X_train,
                y_train,
                X_test,
                y_test,
                models,
                target_weighter,
                encoder,
                reweighter,
                report_name=f"{args.out_path}/{timestamp}/{data_name}/"
                f"_{__get_class_name(reweighter)}",
                data_name=data_name,
                n_iter=args.n_iter,
            )

            report.make_report()
        except Exception as e:
            logger.error(f"Error during computing {data_name}: {e}")
            traceback.print_exc()

    auto_summary_report = AutoSummaryReport(f"{args.out_path}/{timestamp}")
    auto_summary_report.make_report()


if __name__ == "__main__":
    main()
