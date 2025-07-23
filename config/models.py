from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from deslib.dcs.lca import LCA
from deslib.dcs.ola import OLA
from deslib.des.meta_des import METADES
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from deslib.des.knora_u import KNORAU
from functools import partial

MODELS_LISTS = {
    "SIMPLE-1": [
        partial(LogisticRegression, max_iter=1_000),
        GaussianNB,
        DecisionTreeClassifier,
        RandomForestClassifier,
        KNeighborsClassifier,
    ]
}

BASELINES_LIST = {
    "LCA": LCA,
    "OLA": OLA,
    "METADES": METADES,
    "KNORAU": KNORAU,
    "stacking": partial(
        StackingClassifier, final_estimator=LogisticRegression()
    ),
}
