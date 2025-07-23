from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

MODELS_LISTS = {
    "SIMPLE-1": [
        partial(LogisticRegression, max_iter=1_000),
        GaussianNB,
        DecisionTreeClassifier,
        RandomForestClassifier,
        KNeighborsClassifier,
    ]
}
