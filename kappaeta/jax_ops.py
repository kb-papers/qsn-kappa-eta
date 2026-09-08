import jax
import jax.numpy as jnp
from jax import grad, jit


@jit
def kappa_encode_jax(X_eval, X_train_col, y_train, kappa):
    """Encode a single feature column via kappa-weighted imputation (JAX).

    Uses log-space arithmetic for numerical stability with large kappa.

    Parameters
    ----------
    X_eval : jnp.ndarray, shape (n_eval,)
        Test feature column values.
    X_train_col : jnp.ndarray, shape (n_train,)
        Training feature column values.
    y_train : jnp.ndarray, shape (n_train,)
        Training targets.
    kappa : float
        Decay exponent.

    Returns
    -------
    jnp.ndarray, shape (n_eval,)
        Encoded feature values.
    """
    distances = jnp.abs(X_eval[:, None] - X_train_col[None, :])
    log_weights = -kappa * jnp.log1p(distances)
    log_weights_stable = log_weights - jnp.max(log_weights, axis=1, keepdims=True)
    weights = jnp.exp(log_weights_stable)

    numerator = jnp.sum(y_train[None, :] * weights, axis=1)
    denominator = jnp.sum(weights, axis=1) + 1e-12
    return numerator / denominator


@jit
def eta_predict_jax(X_eval, X_train, y_train, eta):
    """Predict using eta-weighted distance regression (JAX).

    Uses log-space weight computation for numerical stability with large eta.

    Parameters
    ----------
    X_eval : jnp.ndarray, shape (n_eval, n_features)
        Encoded evaluation samples.
    X_train : jnp.ndarray, shape (n_train, n_features)
        Encoded training samples.
    y_train : jnp.ndarray, shape (n_train,)
        Training targets.
    eta : float
        Distance-decay exponent.

    Returns
    -------
    jnp.ndarray, shape (n_eval,)
        Predicted values.
    """
    diff = X_eval[:, None, :] - X_train[None, :, :]
    # Small epsilon inside sqrt avoids NaN gradient at distance=0
    distances = jnp.sqrt(jnp.sum(diff ** 2, axis=2) + 1e-12)

    log_weights = -eta * jnp.log1p(distances)
    log_weights_stable = log_weights - jnp.max(log_weights, axis=1, keepdims=True)
    weights = jnp.exp(log_weights_stable)

    numerator = jnp.sum(y_train[None, :] * weights, axis=1)
    denominator = jnp.sum(weights, axis=1) + 1e-12
    return numerator / denominator


@jit
def pipeline_forward_jax(X_eval, X_train, y_train, kappa, eta):
    """Full forward pass: kappa-encode then eta-predict (JAX-differentiable).

    Parameters
    ----------
    X_eval, X_train : jnp.ndarray
        Feature matrices.
    y_train : jnp.ndarray
        Training targets.
    kappa : jnp.ndarray
        Per-feature kappa values (scalar is broadcast).
    eta : float
        Eta decay exponent.

    Returns
    -------
    jnp.ndarray, shape (n_eval,)
        Predicted values.
    """
    n_features = X_eval.shape[1]
    is_scalar_kappa = jnp.ndim(kappa) == 0

    if is_scalar_kappa:
        kappa_array = jnp.full(n_features, kappa)
    else:
        kappa_array = jnp.reshape(kappa, (n_features,))

    # Encode the evaluation set
    def encode_eval_col(p, X_encoded):
        encoded_col = kappa_encode_jax(
            X_eval[:, p], X_train[:, p], y_train, kappa_array[p]
        )
        return X_encoded.at[:, p].set(encoded_col)

    X_eval_encoded = jax.lax.fori_loop(
        0, n_features, encode_eval_col, jnp.zeros_like(X_eval)
    )

    # Encode training data
    def encode_train_col(p, X_encoded):
        encoded_col = kappa_encode_jax(
            X_train[:, p], X_train[:, p], y_train, kappa_array[p]
        )
        return X_encoded.at[:, p].set(encoded_col)

    X_train_encoded = jax.lax.fori_loop(
        0, n_features, encode_train_col, jnp.zeros_like(X_train)
    )

    return eta_predict_jax(X_eval_encoded, X_train_encoded, y_train, eta)


@jit
def mse_loss(y_true, y_pred):
    """Mean squared error."""
    return jnp.mean((y_true - y_pred) ** 2)


@jit
def loss_function_jax(params, X_eval, X_train, y_train, y_eval):
    """MSE loss as a function of ``{'kappa': ..., 'eta': ...}``.

    Parameters
    ----------
    params : dict
        ``{'kappa': jnp.ndarray, 'eta': jnp.ndarray}``.
    X_eval, X_train : jnp.ndarray
        Feature matrices.
    y_train, y_eval : jnp.ndarray
        Target vectors.

    Returns
    -------
    float
        Mean squared error over the evaluation set.
    """
    y_pred = pipeline_forward_jax(
        X_eval, X_train, y_train, params["kappa"], params["eta"]
    )
    return mse_loss(y_eval, y_pred)


grad_loss_jax = jit(grad(loss_function_jax))
loss_and_grad_jax = jit(jax.value_and_grad(loss_function_jax))


def compute_gradients_jax(kappa, eta, X_eval, X_train, y_train, y_eval):
    """Convenience wrapper returning ``(grad_kappa, grad_eta)`` as JAX arrays.

    All inputs are automatically converted to JAX arrays if necessary.
    """
    X_eval_jax = jnp.array(X_eval)
    X_train_jax = jnp.array(X_train)
    y_train_jax = jnp.array(y_train)
    y_eval_jax = jnp.array(y_eval)

    params = {"kappa": jnp.array(kappa), "eta": jnp.array(eta)}
    grads = grad_loss_jax(params, X_eval_jax, X_train_jax, y_train_jax, y_eval_jax)
    return grads["kappa"], grads["eta"]


def compute_loss_and_grad_jax(kappa, eta, X_eval, X_train, y_train, y_eval):
    """Convenience wrapper returning ``(loss, grad_kappa, grad_eta)`` in one traced call.

    Reuses the forward pass that autodiff already computes for the gradient,
    instead of a caller running ``loss_function_jax`` a second time through
    the O(n_train * n_eval) encode/predict pipeline just to log the loss.
    """
    X_eval_jax = jnp.array(X_eval)
    X_train_jax = jnp.array(X_train)
    y_train_jax = jnp.array(y_train)
    y_eval_jax = jnp.array(y_eval)

    params = {"kappa": jnp.array(kappa), "eta": jnp.array(eta)}
    loss, grads = loss_and_grad_jax(params, X_eval_jax, X_train_jax, y_train_jax, y_eval_jax)
    return loss, grads["kappa"], grads["eta"]
