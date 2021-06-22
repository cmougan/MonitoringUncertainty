import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES


class DistributionShift(BaseEstimator, TransformerMixin):
    """
    Transformer implements several distribution shift transformations.
    Example
    -------
    import numpy as np
    import pandas as pd

    from distributions import DistributionShift


    X = pd.DataFrame(
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), columns=["a", "b", "c"]
        )

    y = np.array([0, 1.0, 2.0])


    # Param = 1
    db = DistributionShift(param=1.0)
    db.fit(X, y)
    X_shifted = db.transform(X)
    """

    def __init__(self, param: float = 0, cols=[], strategy="covariateShift"):
        """

        :param data:
        :param columns: If columns is empty all columns are transformed
        """

        if strategy not in ("covariateShift"):
            raise ValueError(
                "Known distribution shifts are 'covariateShift'. Got %s." % strategy
            )
        self.strategy = strategy

        # Columns to transform
        self.cols = cols

        if not isinstance(param, (float, int)):
            raise ValueError("param should be a float or int")
        self.param = param

        # List of dic with stats values to aggregate
        self.mapping_ = []

        # Assume there is no need to convert to dataframe
        self.convert_dataframe = False

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, "scale_"):
            del self.mapping_
            del self.cols
            del self.columns_transform

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.
            .. versionadded:: 0.24
               parameter *sample_weight* support to StandardScaler.
        Returns
        -------
        self : object
            Fitted scaler.
        """

        # Reset internal state before fitting
        self._reset()

        if sparse.issparse(X):
            raise TypeError("DistributionShift does not support sparse input. ")

        # Convert to DataFrame in case is np.ndarray
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            self.convert_dataframe = True

        # If no columns are indicated uses all
        if len(self.cols) < 1:
            self.cols = X.columns

        mapping = []
        for col in self.cols:
            mapping.append(
                {
                    "col": col,
                    "min": np.min(X[col]),
                    "max": np.max(X[col]),
                    "std": np.std(X[col]),
                }
            )
        self.mapping_ = mapping

        return self

    def transform(self, X, parameter=None, columns_transform=[]):
        check_is_fitted(self)

        if self.convert_dataframe:
            X = pd.DataFrame(X)

        if parameter == 0:
            return X
        elif parameter != None:
            self.param = parameter

        if columns_transform != []:
            self.cols = columns_transform

        # Sanity check on the columns
        # self._check_columns_transform(self.cols, columns_transform)

        if self.strategy == "covariateShift":
            return self.transform_covariateShif(X)
        else:
            raise ValueError("Distribution Shift Strategy not supported")

    def transform_covariateShif(self, X):
        Xt = X.copy()

        # For every mapped column (mapped columns are selected in the fit)
        for item in self.mapping_:
            # Only transform those specified on the transform
            if item["col"] in self.cols:
                Xt[item["col"]] = X[item["col"]] + self.param * item["std"]

        # Convert back to numpy array
        if self.convert_dataframe:
            Xt = Xt.to_numpy()

        return Xt

    def _check_subset(self, L1, L2):
        S1 = set(L1)
        S2 = set(L2)

        # Check for repeated values
        if len(S1) != len(L1):
            raise ValueError("Column in the fit are repeated")
        if len(S2) != len(L2):
            raise ValueError("Column in the transform are repeated")

        if S1.intersection(S2) != S1:
            raise ValueError("Column for transform were excluded in the fit.")
        return ""
