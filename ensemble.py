import numpy as np
from sklearn.ensemble import RandomForestClassifier


class StdForestClassifier:
    """Random forest with median aggregation

    Very similar to random forest regressor, but aggregating using the median
    instead of the mean. Can improve the mean absolute error a little.

    Example
    -------
    >>> from sktools import MedianForestRegressor
    >>> from sklearn.datasets import load_boston
    >>> boston = load_boston()['data']
    >>> y = load_boston()['target']
    >>> mf = MedianForestRegressor()
    >>> mf.fit(boston, y)
    >>> mf.predict(boston)[0:10]
    array([24. , 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, 18.9])

    """

    def __init__(self, *args, **kwargs):

        self.rf = RandomForestClassifier(*args, **kwargs)

    def fit(self, X, y):

        self.rf.fit(X, y)

        return self

    def predict_proba(self, X):
        tree_predictions = [tree.predict_proba(X) for tree in self.rf.estimators_]
        return np.mean(np.array(tree_predictions), axis=0)

    def predict(self, X):
        tree_predictions = [tree.predict(X) for tree in self.rf.estimators_]
        return np.median(np.array(tree_predictions), axis=0)

    def predict_std(self, X):
        tree_predictions = [tree.predict_proba(X) for tree in self.rf.estimators_]
        return np.std(np.array(tree_predictions), axis=0)
