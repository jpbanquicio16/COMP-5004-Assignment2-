"""
numcompute.preprocessing


Scalers and OneHotEncoder with fit/transform API.

Conventions

- All inputs converted to float ndarray via np.asarray.
- Scalers do not change shape: X (n,m) -> X_out (n,m)
- OneHotEncoder changes shape: X (n,) -> (n, n_categories)
- Axis 0 = per column (default), following numpy standard.
- NaN behaviour: NaN propagates, mean/std ignores? here we propagate.

 Section 1 - preprocessing
"""

import numpy as np

__all__ = ["StandardScaler", "MinMaxScaler", "OneHotEncoder"]

class StandardScaler:
    """
    Standardize to mean 0, std 1.

    Parameters

    None

    Attributes

    mean_ : ndarray, shape (n_features,)
    scale_ : ndarray, shape (n_features,)
    n_features_in_ : int

    Notes

    Time O(n*m), Space O(m) for mean/std
    Input shape (n_samples, n_features), Output same shape
    NaN: if X has NaN, result will have NaN (propagates)
    """

    def __init__(self):
        # init with None, will be set in fit
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """
        Learn mean and std.

        Parameters

        X : array_like, shape (n_samples, n_features) or (n_samples,)

        Returns

        self

        Raises

        ValueError if X empty

        Notes

        Time O(n*m), Space O(m)
        """
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            raise ValueError("X is empty")
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # mean per column, vectorized
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # avoid divide by zero
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """
        Apply standardization.

        Parameters

        X : array_like, shape (n_samples, n_features)

        Returns

        ndarray, shape (n_samples, n_features) - same as input

        Raises

        ValueError if fit not called before

        Notes

        Time O(n*m), Space O(n*m) for output (copy, not view)
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Must call fit before transform")

        X = np.asarray(X, dtype=float)
        is_1d = False
        if X.ndim == 1:
            is_1d = True
            X = X.reshape(-1, 1)

        # vectorized, no loops
        result = (X - self.mean_) / self.scale_

        if is_1d:
            return result.ravel()
        return result

    def fit_transform(self, X):
        """Fit and transform in one step, same as fit().transform()"""
        return self.fit(X).transform(X)

class MinMaxScaler:
    """
    Scale to given feature_range, default (0,1).

    Parameters

    feature_range : tuple (min, max), default (0,1)

    Notes

    Input shape (n,m) -> Output shape (n,m) same
    Time O(n*m), Space O(m)
    NaN propagates
    """

    def __init__(self, feature_range=(0, 1)):
        # check feature_range
        if len(feature_range)!= 2:
            raise ValueError("feature_range must be (min, max)")
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """
        Learn min and max.

        Parameters

        X : array_like, shape (n_samples, n_features)

        Returns

        self

        Raises

        ValueError if empty
        """
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            raise ValueError("empty X")
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.min_ = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        self.scale_ = max_val - self.min_
        # avoid zero division
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """
        Scale to feature_range.

        Parameters

        X : array_like, shape (n_samples, n_features)

        Returns

        ndarray, same shape as X

        Raises

        ValueError if not fit

        Notes

        Time O(n*m), Space O(n*m) copy
        """
        if self.min_ is None:
            raise ValueError("call fit first")

        X = np.asarray(X, dtype=float)
        is_1d = False
        if X.ndim == 1:
            is_1d = True
            X = X.reshape(-1, 1)

        X_std = (X - self.min_) / self.scale_
        min_r, max_r = self.feature_range
        result = X_std * (max_r - min_r) + min_r

        if is_1d:
            return result.ravel()
        return result

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class OneHotEncoder:
    """
    One hot encode categories.

    Notes

    Input shape (n,) -> Output shape (n, n_categories) - changes shape!
    Time O(n*k), Space O(n*k)
    NaN: treated as category if present
    """

    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        """
        Find unique categories.

        Parameters

        X : array_like, shape (n,)

        Returns

        self
        """
        X = np.asarray(X)
        if X.size == 0:
            raise ValueError("empty X")
        self.categories_ = np.unique(X)
        return self

    def transform(self, X):
        """
        One hot encode.

        Parameters

        X : array_like, shape (n,)

        Returns

        ndarray, shape (n_samples, n_categories)

        Raises

        ValueError if not fit
        """
        if self.categories_ is None:
            raise ValueError("must fit first")

        X = np.asarray(X).ravel()
        # vectorized compare, no loop
        result = (X[:, None] == self.categories_[None, :]).astype(int)
        return result

    def fit_transform(self, X):
        return self.fit(X).transform(X)
