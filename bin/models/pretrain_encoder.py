from datetime import datetime
from pathlib import Path

from liltab.data.dataloaders import (
    ComposedDataLoader,
    FewShotDataLoader,
    RepeatableOutputComposedDataLoader,
)
from liltab.data.datasets import AdaptiveeDataset
from liltab.data.factory import ComposedDataLoaderFactory
from liltab.model.heterogenous_attributes_network import (
    HeterogenousAttributesNetwork,
)
from liltab.train.trainer import HeterogenousAttributesNetworkTrainer
from torch import nn
from bin.models.args import get_pretrain_encoder_args as get_args
from config.models import MODELS_LISTS
from time import time
from loguru import logger


def main():

    args = get_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    n_models = len(MODELS_LISTS[args.model_list_id])

    train_loader = (
        ComposedDataLoaderFactory.create_composed_dataloader_from_path(
            path=Path(args.train_path),
            dataset_cls=AdaptiveeDataset,
            dataset_creation_args={
                "model_id": args.model_list_id,
                "alpha": args.alpha_name,
            },
            loader_cls=FewShotDataLoader,
            dataloader_creation_args={
                "support_size": args.support_size,
                "query_size": 0,
            },
            composed_dataloader_cls=RepeatableOutputComposedDataLoader,
            batch_size=args.batch_size,
        )
    )

    val_loader = (
        ComposedDataLoaderFactory.create_composed_dataloader_from_path(
            path=Path(args.val_path),
            dataset_cls=AdaptiveeDataset,
            dataset_creation_args={
                "model_id": args.model_list_id,
                "alpha": args.alpha_name,
            },
            loader_cls=FewShotDataLoader,
            dataloader_creation_args={
                "support_size": args.support_size,
                "query_size": 0,
            },
            composed_dataloader_cls=ComposedDataLoader,
            batch_size=args.batch_size,
        )
    )

    model = HeterogenousAttributesNetwork(
        hidden_representation_size=n_models,
        n_hidden_layers=args.n_hidden_layers,
        hidden_size=args.hidden_size,
        dropout_rate=args.dropout_rate,
        inner_activation_function=nn.ReLU(),
        output_activation_function=nn.Softmax(),
        is_classifier=False,
    )

    trainer = HeterogenousAttributesNetworkTrainer(
        n_epochs=args.n_epochs,
        gradient_clipping=False,
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_intervals=100,
        file_logger=True,
        tb_logger=True,
        model_checkpoints=True,
        results_path=Path(args.out_path),
    )

    start = time()
    trainer.pretrain_adaptivee(
        model=model, train_loader=train_loader, val_loader=val_loader
    )
    logger.info(f"TRaining time: {time() - start}s")

    trainer.trainer.save_checkpoint(f"resources/models/model_{timestamp}.ckpt")


if __name__ == "__main__":
    main()
