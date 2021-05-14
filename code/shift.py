import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class DistributionShift(BaseEstimator, TransformerMixin):
    """
    Transformer implements several distribution shift transformations.
    """

    def __init__(self, cols=[], strategy="covariateShift"):
        """

        :param data:
        :param columns: If columns is empty all columns are transformed
        """

        if strategy not in ("covariateShift"):
            raise ValueError(
                "Known distribution shifts are 'covariateShift'. Got %s." % strategy
            )

        # If no columns are indicated uses all

        if len(cols) > 1:
            self.cols = cols
        else:
            self.cols = []

    def fit(self, X):
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.X
