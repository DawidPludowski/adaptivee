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
    create_circles,
    create_cubes,
    create_linear,
    create_normal_distribution,
)

DATASETS: list[tuple[str, tuple[np.ndarray, np.ndarray]]] = [
    ("cricles-simple", create_circles()),
    ("linear-simple", create_linear()),
    ("cubes-simple", create_cubes(n_cubes=5)),
    ("normal-dist-simple", create_normal_distribution()),
]

MODELS: list[tuple[any]] = [
    (
        LogisticRegression,
        DecisionTreeClassifier,
        SVC,
        LinearDiscriminantAnalysis,
    )
]

ENCODERS: list[MixInEncoder] = [partial(NLPEncoder, [100]), DummyEncoder]
REWEIGHTERS: list[MixInReweight] = [SimpleReweight]
TARGET_WEIGHTERS: list[MixInTargetWeighter] = [
    SoftMaxWeighter,
    StaticGridWeighter,
]
