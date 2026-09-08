"""MLX (Apple GPU) equivalents of jax_ops.py's kappa/eta pipeline and loss/gradient.

This is the backend the paper's optimizer experiments run on.
"""

import mlx.core as mx
import numpy as np

_DEFAULT_TARGET_BYTES = 256 * 1024 * 1024  # 256MB budget per pairwise chunk tensor
_MIN_BLOCK = 64
_MAX_BLOCK = 4096
_DTYPE_BYTES = 4  # float32


def _default_block_size(n_query, n_key, feature_multiplier=1, target_bytes=_DEFAULT_TARGET_BYTES):
    """Pick a query-axis chunk size so a (block, n_key[, feature_multiplier])
    float32 tensor stays under ``target_bytes``.

    ``feature_multiplier`` must reflect the size of the *other* axes of the
    pairwise tensor (e.g. n_features for eta_predict's (block, n_key, n_features)
    tensor) -- sizing off n_key alone is not enough and can still OOM at scale.
    """
    denom = max(1, n_key * feature_multiplier * _DTYPE_BYTES)
    block = target_bytes // denom
    block = max(_MIN_BLOCK, min(_MAX_BLOCK, block))
    return int(min(block, max(n_query, 1)))


def _as_mx(x, dtype=mx.float32):
    if isinstance(x, mx.array):
        return x.astype(dtype) if x.dtype != dtype else x
    return mx.array(np.asarray(x), dtype=dtype)


def _as_mx_scalar(x):
    return _as_mx(x)


@mx.compile
def _kappa_encode_chunk_kernel(X_query_chunk, X_train_col, y_train, kappa):
    distances = mx.abs(X_query_chunk[:, None] - X_train_col[None, :])
    log_weights = -kappa * mx.log1p(distances)
    log_weights_stable = log_weights - mx.max(log_weights, axis=1, keepdims=True)
    weights = mx.exp(log_weights_stable)

    numerator = mx.sum(y_train[None, :] * weights, axis=1)
    denominator = mx.sum(weights, axis=1) + 1e-12
    return numerator / denominator


def kappa_encode_mlx(X_query, X_train_col, y_train, kappa, block_size=None):
    """Encode a single feature column via kappa-weighted imputation (MLX).

    Chunked over the query axis to bound peak memory; mirrors
    ``jax_ops.kappa_encode_jax`` numerically (log-space stable weights).

    Parameters
    ----------
    X_query : array-like, shape (n_query,)
        Values to encode (evaluation or reference column).
    X_train_col : array-like, shape (n_train,)
        Training feature column values.
    y_train : array-like, shape (n_train,)
        Training targets.
    kappa : float or mx.array
        Decay exponent.
    block_size : int, optional
        Query-axis chunk size. Defaults to :func:`_default_block_size`.

    Returns
    -------
    mx.array, shape (n_query,)
    """
    X_query_mx = _as_mx(X_query)
    X_train_col_mx = _as_mx(X_train_col)
    y_train_mx = _as_mx(y_train)
    kappa_mx = _as_mx_scalar(kappa)

    n_query = X_query_mx.shape[0]
    n_key = X_train_col_mx.shape[0]
    block = block_size if block_size is not None else _default_block_size(n_query, n_key, feature_multiplier=1)

    if n_query <= block:
        return _kappa_encode_chunk_kernel(X_query_mx, X_train_col_mx, y_train_mx, kappa_mx)

    outputs = []
    for start in range(0, n_query, block):
        end = min(start + block, n_query)
        chunk = _kappa_encode_chunk_kernel(X_query_mx[start:end], X_train_col_mx, y_train_mx, kappa_mx)
        mx.eval(chunk)
        outputs.append(chunk)
    return mx.concatenate(outputs)


@mx.compile
def _eta_predict_chunk_kernel(X_query_chunk, X_train, y_train, eta):
    diff = X_query_chunk[:, None, :] - X_train[None, :, :]
    # Small epsilon inside sqrt avoids NaN gradient at distance=0
    distances = mx.sqrt(mx.sum(diff ** 2, axis=2) + 1e-12)

    log_weights = -eta * mx.log1p(distances)
    log_weights_stable = log_weights - mx.max(log_weights, axis=1, keepdims=True)
    weights = mx.exp(log_weights_stable)

    numerator = mx.sum(y_train[None, :] * weights, axis=1)
    denominator = mx.sum(weights, axis=1) + 1e-12
    return numerator / denominator


def eta_predict_mlx(X_query, X_train, y_train, eta, block_size=None):
    """Predict using eta-weighted distance regression (MLX).

    Chunked over the query axis to bound peak memory; mirrors
    ``jax_ops.eta_predict_jax`` numerically (log-space stable weights).

    Parameters
    ----------
    X_query : array-like, shape (n_query, n_features)
        Encoded query samples.
    X_train : array-like, shape (n_train, n_features)
        Encoded training samples.
    y_train : array-like, shape (n_train,)
        Training targets.
    eta : float or mx.array
        Distance-decay exponent.
    block_size : int, optional
        Query-axis chunk size. Defaults to :func:`_default_block_size`
        (sized off n_train * n_features, not just n_train).

    Returns
    -------
    mx.array, shape (n_query,)
    """
    X_query_mx = _as_mx(X_query)
    X_train_mx = _as_mx(X_train)
    y_train_mx = _as_mx(y_train)
    eta_mx = _as_mx_scalar(eta)

    n_query = X_query_mx.shape[0]
    n_key = X_train_mx.shape[0]
    n_features = X_train_mx.shape[1]
    block = (
        block_size
        if block_size is not None
        else _default_block_size(n_query, n_key, feature_multiplier=n_features)
    )

    if n_query <= block:
        return _eta_predict_chunk_kernel(X_query_mx, X_train_mx, y_train_mx, eta_mx)

    outputs = []
    for start in range(0, n_query, block):
        end = min(start + block, n_query)
        chunk = _eta_predict_chunk_kernel(X_query_mx[start:end], X_train_mx, y_train_mx, eta_mx)
        mx.eval(chunk)
        outputs.append(chunk)
    return mx.concatenate(outputs)


def pipeline_forward_mlx(X_eval, X_train, y_train, kappa, eta, block_size_encode=None, block_size_predict=None):
    """Full forward pass: kappa-encode then eta-predict (MLX, differentiable).

    Mirrors ``jax_ops.pipeline_forward_jax``.

    Parameters
    ----------
    X_eval, X_train : array-like
        Feature matrices.
    y_train : array-like
        Training targets.
    kappa : float or array-like
        Per-feature kappa values (scalar is broadcast).
    eta : float
        Eta decay exponent.

    Returns
    -------
    mx.array, shape (n_eval,)
    """
    X_eval_mx = _as_mx(X_eval)
    X_train_mx = _as_mx(X_train)
    y_train_mx = _as_mx(y_train)
    n_features = X_eval_mx.shape[1]

    kappa_mx = _as_mx_scalar(kappa)
    if kappa_mx.ndim == 0:
        # NOTE: mx.full here (instead of broadcast_to) miscomputes gradients when
        # its output is sliced and fed into an @mx.compile'd function multiple
        # times within one mx.grad trace -- confirmed as an mx.compile/mx.full
        # interaction bug (repro: mx.full + per-index calls into a compiled
        # function overcounts the accumulated gradient). broadcast_to is safe.
        kappa_array = mx.broadcast_to(kappa_mx, (n_features,))
    else:
        kappa_array = mx.reshape(kappa_mx, (n_features,))

    eta_mx = _as_mx_scalar(eta)

    eval_cols = []
    train_cols = []
    for p in range(n_features):
        kp = kappa_array[p]
        eval_cols.append(
            kappa_encode_mlx(X_eval_mx[:, p], X_train_mx[:, p], y_train_mx, kp, block_size=block_size_encode)
        )
        train_cols.append(
            kappa_encode_mlx(X_train_mx[:, p], X_train_mx[:, p], y_train_mx, kp, block_size=block_size_encode)
        )

    X_eval_encoded = mx.stack(eval_cols, axis=1)
    X_train_encoded = mx.stack(train_cols, axis=1)

    return eta_predict_mlx(X_eval_encoded, X_train_encoded, y_train_mx, eta_mx, block_size=block_size_predict)


def mse_loss_mlx(y_true, y_pred):
    """Mean squared error."""
    return mx.mean((y_true - y_pred) ** 2)


def loss_function_mlx(params, X_eval, X_train, y_train, y_eval):
    """MSE loss as a function of ``{'kappa': ..., 'eta': ...}``.

    Mirrors ``jax_ops.loss_function_jax``. Assumes inputs are already
    mx.array (matching jax_ops's convention of assuming already-jnp arrays);
    conversion happens in ``compute_gradients_mlx``.
    """
    y_pred = pipeline_forward_mlx(X_eval, X_train, y_train, params["kappa"], params["eta"])
    return mse_loss_mlx(y_eval, y_pred)


grad_loss_mlx = mx.grad(loss_function_mlx)
_grad_loss_mlx_compiled = mx.compile(grad_loss_mlx)

# Same value+grad fusion as jax_ops.loss_and_grad_jax: computes the loss and
# gradient from one forward pass instead of two.
loss_and_grad_mlx = mx.value_and_grad(loss_function_mlx)
_loss_and_grad_mlx_compiled = mx.compile(loss_and_grad_mlx)


def compute_gradients_mlx(kappa, eta, X_eval, X_train, y_train, y_eval):
    """Convenience wrapper returning ``(grad_kappa, grad_eta)`` as MLX arrays.

    All inputs are automatically converted to MLX arrays if necessary.
    """
    X_eval_mx = _as_mx(X_eval)
    X_train_mx = _as_mx(X_train)
    y_train_mx = _as_mx(y_train)
    y_eval_mx = _as_mx(y_eval)

    params = {"kappa": _as_mx_scalar(kappa), "eta": _as_mx_scalar(eta)}
    try:
        grads = _grad_loss_mlx_compiled(params, X_eval_mx, X_train_mx, y_train_mx, y_eval_mx)
        mx.eval(grads["kappa"], grads["eta"])
    except ValueError:
        grads = grad_loss_mlx(params, X_eval_mx, X_train_mx, y_train_mx, y_eval_mx)
    return grads["kappa"], grads["eta"]


def compute_loss_and_grad_mlx(kappa, eta, X_eval, X_train, y_train, y_eval):
    """Convenience wrapper returning ``(loss, grad_kappa, grad_eta)`` in one traced call.

    Reuses the forward pass that autodiff already computes for the gradient,
    instead of a caller running ``loss_function_mlx`` a second time through
    the O(n_train * n_eval) encode/predict pipeline just to log the loss.
    """
    X_eval_mx = _as_mx(X_eval)
    X_train_mx = _as_mx(X_train)
    y_train_mx = _as_mx(y_train)
    y_eval_mx = _as_mx(y_eval)

    params = {"kappa": _as_mx_scalar(kappa), "eta": _as_mx_scalar(eta)}
    try:
        loss, grads = _loss_and_grad_mlx_compiled(params, X_eval_mx, X_train_mx, y_train_mx, y_eval_mx)
        mx.eval(loss, grads["kappa"], grads["eta"])
    except ValueError:
        loss, grads = loss_and_grad_mlx(params, X_eval_mx, X_train_mx, y_train_mx, y_eval_mx)
    return loss, grads["kappa"], grads["eta"]


def encode_columns_mlx(X_encoded, columns, train_col_values, train_target_values, kappa_values, block_size=None):
    """Encode feature columns with kappa-weighted imputation (MLX).

    Estimator-facing convenience function mirroring ``_numba.encode_columns``'s
    signature, so ``KappaEncoder``'s MLX branch is a near copy of its numba
    branch. Unlike ``_numba.encode_columns`` (in-place), this returns a new
    array (MLX arrays are immutable).

    Parameters
    ----------
    X_encoded : array-like, shape (n_samples, n_features)
        Data matrix to encode.
    columns : sequence of int, shape (n_features,)
        Column indices to encode.
    train_col_values : sequence of array-like, shape (n_features, n_train)
        Per-column training feature values.
    train_target_values : array-like, shape (n_train,)
        Training target values.
    kappa_values : sequence of float, shape (n_features,)
        Per-column kappa exponents.

    Returns
    -------
    mx.array, shape (n_samples, n_features)
    """
    X_encoded_mx = _as_mx(X_encoded)
    train_target_values_mx = _as_mx(train_target_values)
    n_total_cols = X_encoded_mx.shape[1]

    encoded_by_col = {}
    for idx, col in enumerate(columns):
        col = int(col)
        train_col_mx = _as_mx(train_col_values[idx])
        encoded_by_col[col] = kappa_encode_mlx(
            X_encoded_mx[:, col],
            train_col_mx,
            train_target_values_mx,
            kappa_values[idx],
            block_size=block_size,
        )

    out_cols = [
        encoded_by_col[c] if c in encoded_by_col else X_encoded_mx[:, c] for c in range(n_total_cols)
    ]
    return mx.stack(out_cols, axis=1)
