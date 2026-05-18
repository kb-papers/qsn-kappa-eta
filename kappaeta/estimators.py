import logging

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ._numba import encode_columns, eta_predict_chunk

logger = logging.getLogger(__name__)


class KappaEncoder(BaseEstimator, TransformerMixin):
    """Feature encoder that imputes each column using distance-weighted kappa decay.

    Parameters
    ----------
    kappa : float or array-like
        Decay exponent(s).  A scalar is broadcast to all features;
        an array must have length ``n_features``.
    """

    def __init__(self, kappa=0):
        self.kappa = kappa

    def fit(self, X, y):
        self.columns = list(range(X.shape[1]))
        self.train_col_values = {col: X[:, col].astype(np.float32) for col in self.columns}
        self.train_target_values = y.astype(np.float32)

        # Normalise kappa to a per-column dict
        kappa_val = float(self.kappa) if hasattr(self.kappa, "ndim") else self.kappa
        if isinstance(kappa_val, (int, float)):
            self.kappa = {col: kappa_val for col in self.columns}
        else:
            self.kappa = {col: kappa_val[col] for col in self.columns}

        return self

    def transform(self, X, y=None):
        return self._encode_numba(X.copy())

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def _encode_numba(self, X):
        train_col_values_arr = np.array(
            [self.train_col_values[col] for col in self.columns]
        )
        kappa_values_arr = np.array([self.kappa[col] for col in self.columns])
        return encode_columns(
            X,
            np.array(self.columns),
            train_col_values_arr,
            self.train_target_values,
            kappa_values_arr,
        )


class EtaRegressor(BaseEstimator, RegressorMixin):
    """Distance-weighted regressor using eta-decay kernels.

    Parameters
    ----------
    eta : float
        Distance-decay exponent.
    """

    def __init__(self, eta=1.0):
        self.eta = float(eta) if hasattr(eta, "ndim") else eta

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float32)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=np.float32)

        if self.eta == 0:
            return np.full(X.shape[0], np.mean(self.y_train_))

        return np.asarray(
            eta_predict_chunk(X, self.X_train_, self.y_train_, np.float32(self.eta))
        )


class KappaEtaRegressor(BaseEstimator, RegressorMixin):
    """End-to-end regressor: MinMaxScaler -> KappaEncoder -> EtaRegressor.

    Parameters
    ----------
    kappa : float
        Kappa decay exponent.
    eta : float
        Eta distance-decay exponent.
    """

    def __init__(self, kappa=2.0, eta=2.0):
        self.kappa = float(kappa) if hasattr(kappa, "ndim") else kappa
        self.eta = float(eta) if hasattr(eta, "ndim") else eta

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float32)
        self.pipeline_ = Pipeline([
            ("scaler", MinMaxScaler()),
            ("encoder", KappaEncoder(kappa=self.kappa)),
            ("regressor", EtaRegressor(eta=self.eta)),
        ])
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "pipeline_")
        X = check_array(X, dtype=np.float32)
        return self.pipeline_.predict(X)

    def grid_search(self, X, y, param_grid=None, cv=5,
                    scoring="neg_mean_squared_error", n_jobs=-1, verbose=1):
        """Exhaustive grid search over kappa/eta values."""
        from sklearn.model_selection import GridSearchCV

        X, y = check_X_y(X, y, dtype=np.float32)
        if param_grid is None:
            param_grid = {
                "kappa": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                "eta": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            }
        gs = GridSearchCV(
            estimator=self,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            refit=True,
        )
        gs.fit(X, y)
        self.best_params_ = gs.best_params_
        self.best_score_ = gs.best_score_
        self.grid_search_results_ = gs.cv_results_
        self.kappa = self.best_params_["kappa"]
        self.eta = self.best_params_["eta"]
        self.fit(X, y)
        return self
