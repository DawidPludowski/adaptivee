from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from adaptivee.encoders import MixInEncoder
from adaptivee.reweighting import MixInReweight
from adaptivee.target_weights import MixInTargetWeighter


class AutoReport:

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: list[any] = None,
        target_weighter: MixInTargetWeighter = None,
        encoder: MixInEncoder = None,
        reweighter: MixInReweight = None,
        meta_data: dict[str, str] = {},
        result_dir: str | Path = "report",
        report_name: str = None,
    ) -> None:
        self.X = X
        self.y = y
        self.models = models
        self.target_weighter = target_weighter
        self.encoder = encoder
        self.reweighter = reweighter

        self.root_dir = Path(result_dir) / report_name
        self.meta_data = meta_data

    def make_report(self):
        self.root_dir.mkdir(exist_ok=True, parents=True)

        self.vizualize_data()

    def vizualize_data(self):

        y = self.y
        if "data name" in self.meta_data.keys():
            title = self.meta_data["data_name"]
        else:
            title = "Data visualization"

        if self.X.shape[1] > 2:
            title += f" (PCA projection from {self.X.shape[1]} dims)"

            pca = PCA()
            X = pca.fit_transform(self.X)
        else:
            X = self.X

        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.title(title)
        plt.savefig(f"{self.root_dir}/data_viz.png")
        plt.clf()
