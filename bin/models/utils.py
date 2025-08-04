import numpy as np


class Oracle:

    def __init__(self, pool_classifiers, invert_oracle):
        self.models = pool_classifiers
        self.invert_oracle = invert_oracle

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X, y_test):
        y_test = y_test.reshape((-1, 1))
        ys = []
        for model in self.models:
            y = model.predict(X).reshape((-1, 1))
            ys.append(y)

        ys = np.hstack(ys)

        diffs = np.abs(ys - y_test)

        if self.invert_oracle:
            best_pred = np.argmax(diffs, axis=1)
        else:
            best_pred = np.argmin(diffs, axis=1)

        return ys[np.arange(ys.shape[0]), best_pred]
