"""CPU vs MLX backend parity, at the ops, estimator and optimizer level.

The paper's Automatic Differentiation section states that the two backends
implement the same log-space computation and are checked against each other;
this is that check.
kappaeta.mlx_ops is compared against kappaeta.jax_ops (forward pass and
gradients) and kappaeta._numba (encode_columns), with a relative tolerance of
~0.1-1% to absorb float32 and GPU-reduction ordering differences.

Skips cleanly (via conftest's module-level importorskip) on machines without
MLX, which is Apple Silicon only.
"""

import jax.numpy as jnp
import mlx.core as mx
import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

from kappaeta import (
    AdamOptimizer,
    EtaRegressor,
    KappaEncoder,
    KappaEtaRegressor,
    LBFGSOptimizer,
    jax_ops,
    mlx_ops,
)
from kappaeta._numba import encode_columns

RTOL = 1e-2
ATOL = 1e-3


def _assert_close(a, b, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


@pytest.mark.parametrize("kappa", [0.5, 2.0, 5.0])
def test_kappa_encode_parity(small_dataset, kappa):
    X_train, X_test, y_train, _ = small_dataset

    jax_out = jax_ops.kappa_encode_jax(
        jnp.array(X_test[:, 0]), jnp.array(X_train[:, 0]), jnp.array(y_train), kappa
    )
    mlx_out = mlx_ops.kappa_encode_mlx(X_test[:, 0], X_train[:, 0], y_train, kappa)
    _assert_close(jax_out, mlx_out)


@pytest.mark.parametrize("eta", [0.0, 0.5, 2.0, 5.0])
def test_eta_predict_parity(small_dataset, eta):
    X_train, X_test, y_train, _ = small_dataset

    jax_out = jax_ops.eta_predict_jax(jnp.array(X_test), jnp.array(X_train), jnp.array(y_train), eta)
    mlx_out = mlx_ops.eta_predict_mlx(X_test, X_train, y_train, eta)
    _assert_close(jax_out, mlx_out)


@pytest.mark.parametrize("kappa,eta", [(0.5, 0.5), (2.0, 2.0), (5.0, 1.0)])
def test_pipeline_forward_and_mse_parity(small_dataset, kappa, eta):
    X_train, X_test, y_train, y_test = small_dataset

    jax_pred = jax_ops.pipeline_forward_jax(
        jnp.array(X_test), jnp.array(X_train), jnp.array(y_train), jnp.array(kappa), eta
    )
    mlx_pred = mlx_ops.pipeline_forward_mlx(X_test, X_train, y_train, kappa, eta)
    _assert_close(jax_pred, mlx_pred)

    jax_mse = jax_ops.mse_loss(jnp.array(y_test), jax_pred)
    mlx_mse = mlx_ops.mse_loss_mlx(mx.array(y_test), mlx_pred)
    _assert_close(jax_mse, mlx_mse)


@pytest.mark.parametrize("kappa,eta", [(0.5, 0.5), (2.0, 2.0), (5.0, 1.0)])
def test_gradient_parity_scalar_kappa(small_dataset, kappa, eta):
    X_train, X_test, y_train, y_test = small_dataset

    gk_jax, ge_jax = jax_ops.compute_gradients_jax(kappa, eta, X_test, X_train, y_train, y_test)
    gk_mlx, ge_mlx = mlx_ops.compute_gradients_mlx(kappa, eta, X_test, X_train, y_train, y_test)

    _assert_close(gk_jax, gk_mlx)
    _assert_close(ge_jax, ge_mlx)


def test_gradient_parity_per_feature_kappa(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset
    n_features = X_train.shape[1]
    kappa = np.linspace(0.5, 3.0, n_features).astype(np.float32)
    eta = 1.5

    gk_jax, ge_jax = jax_ops.compute_gradients_jax(kappa, eta, X_test, X_train, y_train, y_test)
    gk_mlx, ge_mlx = mlx_ops.compute_gradients_mlx(kappa, eta, X_test, X_train, y_train, y_test)

    _assert_close(gk_jax, gk_mlx)
    _assert_close(ge_jax, ge_mlx)


def test_encode_columns_mlx_matches_numba(small_dataset):
    X_train, X_test, y_train, _ = small_dataset
    n_features = X_train.shape[1]

    columns = np.arange(n_features)
    train_col_values = np.array([X_train[:, c] for c in columns])
    kappa_values = np.linspace(0.5, 3.0, n_features).astype(np.float32)

    numba_out = encode_columns(X_test.copy(), columns, train_col_values, y_train, kappa_values)
    mlx_out = mlx_ops.encode_columns_mlx(X_test.copy(), columns, train_col_values, y_train, kappa_values)

    _assert_close(numba_out, mlx_out)


@pytest.mark.parametrize("kappa", [0.5, 2.0])
def test_kappa_encoder_estimator_parity(small_dataset, kappa):
    X_train, X_test, y_train, _ = small_dataset

    cpu_enc = KappaEncoder(kappa=kappa, backend="cpu").fit(X_train, y_train)
    mlx_enc = KappaEncoder(kappa=kappa, backend="mlx").fit(X_train, y_train)

    _assert_close(cpu_enc.transform(X_test), mlx_enc.transform(X_test))


@pytest.mark.parametrize("eta", [0.0, 0.5, 2.0])
def test_eta_regressor_estimator_parity(small_dataset, eta):
    X_train, X_test, y_train, _ = small_dataset

    cpu_reg = EtaRegressor(eta=eta, backend="cpu").fit(X_train, y_train)
    mlx_reg = EtaRegressor(eta=eta, backend="mlx").fit(X_train, y_train)

    _assert_close(cpu_reg.predict(X_test), mlx_reg.predict(X_test))


@pytest.mark.parametrize("kappa,eta", [(0.5, 0.5), (2.0, 2.0), (5.0, 1.0)])
def test_kappa_eta_regressor_predict_and_mse_parity(small_dataset, kappa, eta):
    X_train, X_test, y_train, y_test = small_dataset

    cpu_model = KappaEtaRegressor(kappa=kappa, eta=eta, backend="cpu").fit(X_train, y_train)
    mlx_model = KappaEtaRegressor(kappa=kappa, eta=eta, backend="mlx").fit(X_train, y_train)

    cpu_pred = cpu_model.predict(X_test)
    mlx_pred = mlx_model.predict(X_test)
    _assert_close(cpu_pred, mlx_pred)

    cpu_mse = mean_squared_error(y_test, cpu_pred)
    mlx_mse = mean_squared_error(y_test, mlx_pred)
    assert abs(cpu_mse - mlx_mse) / max(abs(cpu_mse), 1e-8) < RTOL


def test_large_dataset_encode_predict_parity(large_dataset):
    """Exercises the chunked MLX path (15k rows, well past the 4096-row block
    cap). Encode/predict only -- gradient computation is not memory-bounded
    by chunking (see kappaeta/mlx_ops.py's module docstring) and is
    intentionally not exercised at this scale.
    """
    X_train, X_test, y_train, y_test = large_dataset
    kappa, eta = 2.0, 2.0

    jax_pred = jax_ops.pipeline_forward_jax(
        jnp.array(X_test), jnp.array(X_train), jnp.array(y_train), jnp.array(kappa), eta
    )
    mlx_pred = mlx_ops.pipeline_forward_mlx(X_test, X_train, y_train, kappa, eta)
    _assert_close(jax_pred, mlx_pred)

    n_features = X_train.shape[1]
    columns = np.arange(n_features)
    train_col_values = np.array([X_train[:, c] for c in columns])
    kappa_values = np.full(n_features, kappa, dtype=np.float32)

    numba_out = encode_columns(X_test.copy(), columns, train_col_values, y_train, kappa_values)
    mlx_out = mlx_ops.encode_columns_mlx(X_test.copy(), columns, train_col_values, y_train, kappa_values)
    _assert_close(numba_out, mlx_out)


def test_large_dataset_estimator_predict_parity(large_dataset):
    """Same as test_large_dataset_encode_predict_parity but through the
    public KappaEtaRegressor API, at the same chunk-forcing scale."""
    X_train, X_test, y_train, y_test = large_dataset
    kappa, eta = 2.0, 2.0

    cpu_model = KappaEtaRegressor(kappa=kappa, eta=eta, backend="cpu").fit(X_train, y_train)
    mlx_model = KappaEtaRegressor(kappa=kappa, eta=eta, backend="mlx").fit(X_train, y_train)

    _assert_close(cpu_model.predict(X_test), mlx_model.predict(X_test))


@pytest.mark.parametrize("optimizer_cls", [AdamOptimizer, LBFGSOptimizer])
def test_optimizer_single_step_parity(small_dataset, optimizer_cls):
    """A single optimizer step is a tight numerical check: it only depends
    on one loss+gradient evaluation at the same starting point, so float32
    accumulation hasn't had a chance to compound across backends yet.
    """
    X_train, X_test, y_train, y_test = small_dataset

    cpu_opt = optimizer_cls(max_iters=1, verbose=False, backend="cpu")
    mlx_opt = optimizer_cls(max_iters=1, verbose=False, backend="mlx")

    k_cpu, e_cpu, _ = cpu_opt.optimize(2.0, 2.0, X_test, X_train, y_train, y_test)
    k_mlx, e_mlx, _ = mlx_opt.optimize(2.0, 2.0, X_test, X_train, y_train, y_test)

    _assert_close(k_cpu, k_mlx)
    _assert_close(e_cpu, e_mlx)


@pytest.mark.parametrize("optimizer_cls", [AdamOptimizer, LBFGSOptimizer])
def test_optimizer_multi_step_smoke(small_dataset, optimizer_cls):
    """Documents (rather than hides) that exact multi-iteration trajectories
    aren't expected to match bit-for-bit across backends under float32
    accumulation -- this is a loose-tolerance check that both backends
    converge toward the same neighborhood and both reduce the loss overall,
    catching gross divergence without demanding exact agreement.
    """
    X_train, X_test, y_train, y_test = small_dataset

    cpu_opt = optimizer_cls(max_iters=20, verbose=False, backend="cpu")
    mlx_opt = optimizer_cls(max_iters=20, verbose=False, backend="mlx")

    k_cpu, e_cpu, hist_cpu = cpu_opt.optimize(2.0, 2.0, X_test, X_train, y_train, y_test)
    k_mlx, e_mlx, hist_mlx = mlx_opt.optimize(2.0, 2.0, X_test, X_train, y_train, y_test)

    assert hist_cpu[-1]["loss"] <= hist_cpu[0]["loss"]
    assert hist_mlx[-1]["loss"] <= hist_mlx[0]["loss"]

    np.testing.assert_allclose(float(k_cpu), float(k_mlx), rtol=0.1, atol=0.1)
    np.testing.assert_allclose(float(e_cpu), float(e_mlx), rtol=0.1, atol=0.1)
    np.testing.assert_allclose(hist_cpu[-1]["loss"], hist_mlx[-1]["loss"], rtol=0.1, atol=0.1)
