import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES


class DistributionShift(BaseEstimator, TransformerMixin):
    """
    Transformer implements several distribution shift transformations.
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

        if not isinstance(param, (float, int)):
            raise ValueError("param should be a float or int")
        self.param = param

        # If no columns are indicated uses all

        if len(cols) > 1:
            self.cols = cols
        else:
            self.cols = []

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, "scale_"):
            del self.scale_
            del self.mean_
            del self.std_
            del self.min_
            del self.max_

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
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """
        Online computation of mean and std on X for later scaling.

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
        if sparse.issparse(X):
            raise TypeError("DistributionShift does not support sparse input. ")

        n_features = X.shape[1]

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)

        return self

    def transform(self, X, parameter=None):
        check_is_fitted(self)

        self.Xt = X.copy()

        if parameter == 0:
            return X
        elif parameter != None:
            self.param = parameter

        if self.strategy == "covariateShift":
            self.Xt += self.param * self.std_
            return self.Xt
        else:
            raise ValueError("Distribution Shift Strategy not supported")
