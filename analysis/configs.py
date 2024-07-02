import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from adaptivee.encoders import MixInEncoder, NLPEncoder
from adaptivee.reweighting import MixInReweight, SimpleReweight
from adaptivee.target_weights import MixInTargetWeighter, SoftMaxWeighter
from analysis.data import (
    create_circles,
    create_cubes,
    create_linear,
    create_normal_distribution,
)

DATASETS: list[tuple[np.ndarray, np.ndarray]] = [
    create_circles(),
    create_linear(),
    create_cubes(n_cubes=5),
    create_normal_distribution(),
]

MODELS: list[tuple[any]] = [
    (
        LogisticRegression(),
        DecisionTreeClassifier(),
        SVC(),
        LinearDiscriminantAnalysis(),
    )
]

ENCODERS: list[MixInEncoder] = [NLPEncoder([5, 100, 100, 2])]
REWEIGHTERS: list[MixInReweight] = [SimpleReweight()]
TARGET_WEIGHTERS: list[MixInTargetWeighter] = [SoftMaxWeighter()]
