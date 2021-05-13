import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ..utils.validation import check_is_fitted


class DistributionShift(BaseEstimator, TransformerMixin):
    """
    Transformer implements several distribution shift transformations.
    """

    def __init__(self, data, cols=[], strategy="covariateShift"):
        """

        :param data:
        :param columns: If columns is empty all columns are transformed
        """

        # Currently only works with DataFrame TODO works with np.array
        assert isinstance(data, pd.DataFrame)

        assert strategy in ["covariateShift"], "Strategy not supported"

        # If no columns are indicated uses all
        self.data = data
        if len(cols) > 1:
            self.cols = cols
        else:
            self.cols = self.data.columns

    def fit(self, X):
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.X
