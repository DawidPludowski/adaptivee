from functools import partial

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from adaptivee.encoders import DummyEncoder, MixInEncoder, NLPEncoder
from adaptivee.reweighting import MixInReweight, SimpleReweight
from adaptivee.target_weights import (
    MixInTargetWeighter,
    SoftMaxWeighter,
    StaticGridWeighter,
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
            change_position(create_circles(), [2, 2]),
            change_position(create_linear(), [-2, -2]),
        ),
    ),
    (
        "cubes-normal-mix",
        mix_data(
            change_position(create_cubes(), [2, 2]),
            change_position(create_normal_distribution(), [-2, -2]),
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

ENCODERS: list[MixInEncoder] = [partial(NLPEncoder, [100, 100]), DummyEncoder]
REWEIGHTERS: list[MixInReweight] = [SimpleReweight]
TARGET_WEIGHTERS: list[MixInTargetWeighter] = [
    SoftMaxWeighter,
    StaticGridWeighter,
]
