from functools import partial

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from adaptivee.encoders import MixInEncoder, NLPEncoder
from adaptivee.reweighting import (
    DirectionReweight,
    MixInReweight,
    SimpleReweight,
)
from adaptivee.target_weights import (
    MixInTargetWeighter,
    SoftMaxWeighter,
    StaticGridWeighter,
    StaticLogisticWeighter,
)
from analysis.data import (
    change_position,
    create_circles,
    create_cubes,
    create_linear,
    create_normal_distribution,
    mix_data,
)

DATASETS: list[tuple[str, tuple[np.ndarray, np.ndarray]]] = [
    (
        "circles-linear-mix",
        mix_data(
            change_position(create_circles(n=10_000, p=6), [1] * 6),
            change_position(create_linear(n=1_000, p=6), [-2] * 6),
        ),
    ),
    (
        "cubes-normal-mix",
        mix_data(
            change_position(create_cubes(n=5_000), [1] * 2),
            change_position(create_normal_distribution(n=5_000), [-1] * 2),
        ),
    ),
]

MODELS: list[tuple[any]] = [
    (
        LogisticRegression,
        DecisionTreeClassifier,
        SVC,
        LinearDiscriminantAnalysis,
    )
]

ENCODERS: list[MixInEncoder] = [partial(NLPEncoder, [100, 100])]
REWEIGHTERS: list[MixInReweight] = [SimpleReweight, DirectionReweight]
TARGET_WEIGHTERS: list[MixInTargetWeighter] = [
    SoftMaxWeighter,
    StaticGridWeighter,
    StaticLogisticWeighter,
]
