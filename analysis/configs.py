from functools import partial

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from adaptivee.encoders import LiltabEncoder, MixInEncoder, MLPEncoder
from adaptivee.reweighting import (
    DirectionReweight,
    MixInReweight,
    SimpleReweight,
)
from adaptivee.target_weights import (
    MixInStaticTargetWeighter,
    MixInTargetWeighter,
    OneHotWeighter,
    SoftMaxWeighter,
    StaticEqualWeighter,
    StaticLogisticWeighter,
)
from analysis.data import (
    blur_data,
    change_position,
    create_circles,
    create_cubes,
    create_linear,
    create_normal_distribution,
    mix_data,
)

DATASETS: list[tuple[str, tuple[np.ndarray, np.ndarray]]] = [
    (
        "circles-linear-mix-blurred",
        blur_data(
            mix_data(
                change_position(create_circles(n=10_000, p=6), [1] * 6),
                change_position(create_linear(n=1_000, p=6), [-2] * 6),
            ),
            magnitude=0.5,
        ),
    ),
    (
        "cubes-normal-mix",
        blur_data(
            mix_data(
                change_position(create_cubes(n=5_000), [1] * 2),
                change_position(create_normal_distribution(n=5_000), [-1] * 2),
            ),
            magnitude=0.2,
        ),
    ),
]

MODELS: list[tuple[any]] = [
    (
        LogisticRegression,
        DecisionTreeClassifier,
        # partial(SVC, probability=True),
        LinearDiscriminantAnalysis,
        GaussianNB,
        RandomForestClassifier,
        KNeighborsClassifier,
    )
]

ENCODERS: list[MixInEncoder] = [
    partial(LiltabEncoder, model_path="resources/models/final_model.ckpt"),
    partial(MLPEncoder, [100, 100]),
]
REWEIGHTERS: list[MixInReweight] = [SimpleReweight, DirectionReweight]
TARGET_WEIGHTERS: list[MixInTargetWeighter] = [
    OneHotWeighter,
    SoftMaxWeighter,
]

STATIC_TARGET_WEIGHTERS: list[MixInStaticTargetWeighter] = [
    StaticLogisticWeighter,
    StaticEqualWeighter,
]
