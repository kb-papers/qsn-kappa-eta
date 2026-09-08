"""scikit-learn estimators for the two kappa-eta stages and their composition.

``KappaEncoder`` is the kappa-encoding stage, ``EtaRegressor`` the eta-prediction
stage, and ``KappaEtaRegressor`` the composed pipeline the paper writes as
``y_hat = g_eta(f_kappa(X))``. These are the forward model only; hyperparameter
selection lives in ``methodology.py``.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ._backend import resolve_backend
from ._numba import encode_columns, eta_predict_chunk

try:
    from . import mlx_ops
except ImportError:
    mlx_ops = None


class KappaEncoder(BaseEstimator, TransformerMixin):
    """Feature encoder that imputes each column using distance-weighted kappa decay.

    Parameters
    ----------
    kappa : float or array-like
        Decay exponent(s).  A scalar is broadcast to all features;
        an array must have length ``n_features``.
    backend : {"mlx", "cpu", None}
        Which implementation to use. ``None`` (default) resolves to MLX if
        installed (Apple Silicon), else "cpu" (numba kernels).
    """

    def __init__(self, kappa=0, backend=None):
        self.kappa = kappa
        self.backend = backend

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

        self.backend_ = resolve_backend(self.backend)
        return self

    def transform(self, X, y=None):
        if self.backend_ == "mlx":
            return self._encode_mlx(X.copy())
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

    def _encode_mlx(self, X):
        train_col_values_arr = [self.train_col_values[col] for col in self.columns]
        kappa_values_arr = [self.kappa[col] for col in self.columns]
        encoded = mlx_ops.encode_columns_mlx(
            X,
            np.array(self.columns),
            train_col_values_arr,
            self.train_target_values,
            kappa_values_arr,
        )
        return np.array(encoded)


class EtaRegressor(BaseEstimator, RegressorMixin):
    """Distance-weighted regressor using eta-decay kernels.

    Parameters
    ----------
    eta : float
        Distance-decay exponent.
    backend : {"mlx", "cpu", None}
        Which implementation to use. ``None`` (default) resolves to MLX if
        installed (Apple Silicon), else "cpu" (numba kernels).
    """

    def __init__(self, eta=1.0, backend=None):
        self.eta = float(eta) if hasattr(eta, "ndim") else eta
        self.backend = backend

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float32)
        self.X_train_ = X
        self.y_train_ = y
        self.backend_ = resolve_backend(self.backend)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=np.float32)

        if self.eta == 0:
            return np.full(X.shape[0], np.mean(self.y_train_))

        if self.backend_ == "mlx":
            return np.array(
                mlx_ops.eta_predict_mlx(X, self.X_train_, self.y_train_, np.float32(self.eta))
            )

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
    backend : {"mlx", "cpu", None}
        Which implementation to use. ``None`` (default) resolves to MLX if
        installed (Apple Silicon), else "cpu" (numba kernels).
    """

    def __init__(self, kappa=2.0, eta=2.0, backend=None):
        self.kappa = float(kappa) if hasattr(kappa, "ndim") else kappa
        self.eta = float(eta) if hasattr(eta, "ndim") else eta
        self.backend = backend

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float32)
        self.pipeline_ = Pipeline([
            ("scaler", MinMaxScaler()),
            ("encoder", KappaEncoder(kappa=self.kappa, backend=self.backend)),
            ("regressor", EtaRegressor(eta=self.eta, backend=self.backend)),
        ])
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "pipeline_")
        X = check_array(X, dtype=np.float32)
        return self.pipeline_.predict(X)
