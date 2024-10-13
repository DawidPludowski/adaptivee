from datetime import datetime
from pathlib import Path

from liltab.data.dataloaders import ComposedDataLoader, FewShotDataLoader
from liltab.data.datasets import PandasDataset
from liltab.data.factory import ComposedDataLoaderFactory
from liltab.model.heterogenous_attributes_network import (
    HeterogenousAttributesNetwork,
)
from liltab.train.trainer import HeterogenousAttributesNetworkTrainer
from torch import nn

DATAPATHS = [
    "SotMaxWeighter-1",
    "SotMaxWeighter-2",
    "SotMaxWeighter-05",
    "OneHotWeighter",
]


def main():

    for datapth in DATAPATHS:

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        train_loader = (
            ComposedDataLoaderFactory.create_composed_dataloader_from_path(
                path=Path("resources/data/openml/encoder-mod/SoftMaxWeighter"),
                dataset_cls=PandasDataset,
                dataset_creation_args={"response_regex": "target.*"},
                loader_cls=FewShotDataLoader,
                dataloader_creation_args={"support_size": 30, "query_size": 0},
                composed_dataloader_cls=ComposedDataLoader,
                batch_size=32,
            )
        )

        model = HeterogenousAttributesNetwork(
            hidden_representation_size=6,
            n_hidden_layers=2,
            hidden_size=8,
            dropout_rate=0.3,
            inner_activation_function=nn.ReLU(),
            output_activation_function=nn.Softmax(),
            is_classifier=False,
        )

        trainer = HeterogenousAttributesNetworkTrainer(
            n_epochs=1000,
            gradient_clipping=False,
            learning_rate=1e-3,
            weight_decay=1e-4,
            early_stopping_intervals=100,
            file_logger=True,
            tb_logger=True,
            model_checkpoints=True,
            results_path=Path("results"),
        )

        trainer.pretrain_adaptivee(model=model, train_loader=train_loader)

        trainer.trainer.save_checkpoint(
            f"resources/models/model_{datapth}_{timestamp}.ckpt"
        )


if __name__ == "__main__":
    main()
