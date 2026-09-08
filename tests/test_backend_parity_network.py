"""Numba vs MLX backend parity over real UCI datasets (network access required).

Deselected by default (see pyproject.toml's `addopts = "-m 'not network'"`);
run explicitly with `pytest -m network` as a manual pre-release check.
"""

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

from kappaeta import EtaRegressor, KappaEncoder, KappaEtaRegressor

RTOL = 1e-2
ATOL = 1e-3

pytestmark = pytest.mark.network


def _assert_close(a, b, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


def test_kappa_encoder_parity_on_uci_dataset(network_dataset):
    X_train, X_test, y_train, _ = network_dataset

    cpu_enc = KappaEncoder(kappa=2.0, backend="cpu").fit(X_train, y_train)
    mlx_enc = KappaEncoder(kappa=2.0, backend="mlx").fit(X_train, y_train)

    _assert_close(cpu_enc.transform(X_test), mlx_enc.transform(X_test))


def test_eta_regressor_parity_on_uci_dataset(network_dataset):
    X_train, X_test, y_train, _ = network_dataset

    cpu_reg = EtaRegressor(eta=2.0, backend="cpu").fit(X_train, y_train)
    mlx_reg = EtaRegressor(eta=2.0, backend="mlx").fit(X_train, y_train)

    _assert_close(cpu_reg.predict(X_test), mlx_reg.predict(X_test))


def test_kappa_eta_regressor_predict_and_mse_parity_on_uci_dataset(network_dataset):
    X_train, X_test, y_train, y_test = network_dataset

    cpu_model = KappaEtaRegressor(kappa=2.0, eta=2.0, backend="cpu").fit(X_train, y_train)
    mlx_model = KappaEtaRegressor(kappa=2.0, eta=2.0, backend="mlx").fit(X_train, y_train)

    cpu_pred = cpu_model.predict(X_test)
    mlx_pred = mlx_model.predict(X_test)
    _assert_close(cpu_pred, mlx_pred)

    cpu_mse = mean_squared_error(y_test, cpu_pred)
    mlx_mse = mean_squared_error(y_test, mlx_pred)
    assert abs(cpu_mse - mlx_mse) / max(abs(cpu_mse), 1e-8) < RTOL
