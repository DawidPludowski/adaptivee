import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def accuracy(y: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = (y_pred > 0.5).astype(int)
    return accuracy_score(y, y_pred)


def balanced_accuracy(y: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = (y_pred > 0.5).astype(int)
    return balanced_accuracy_score(y, y_pred)


def f1(y: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = (y_pred > 0.5).astype(int)
    return f1_score(y, y_pred)
