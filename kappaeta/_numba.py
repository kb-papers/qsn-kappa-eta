import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def kappa_impute_numba(x_t, x_i, y_i, kappa):
    """Weighted imputation of a single test value using kappa decay.

    Parameters
    ----------
    x_t : float
        Test value to impute.
    x_i : ndarray, shape (n_train,)
        Training feature values.
    y_i : ndarray, shape (n_train,)
        Training target values.
    kappa : float
        Decay exponent controlling the distance weighting.

    Returns
    -------
    float
        Imputed value: sum(y * w) / sum(w).
    """
    distances = np.abs(x_t - x_i)
    weights = (1 / ((1 + distances) ** kappa)).astype(np.float32)
    return np.dot(y_i, weights) / np.sum(weights)


@njit(parallel=True)
def encode_columns(X_encoded, columns, train_col_values, train_target_values, kappa_values):
    """Encode all feature columns in-place with kappa-weighted imputation.

    Parameters
    ----------
    X_encoded : ndarray, shape (n_samples, n_features)
        Data matrix to encode (modified in-place).
    columns : ndarray, shape (n_features,)
        Column indices to encode.
    train_col_values : ndarray, shape (n_features, n_train)
        Per-column training feature values.
    train_target_values : ndarray, shape (n_train,)
        Training target values.
    kappa_values : ndarray, shape (n_features,)
        Per-column kappa exponents.

    Returns
    -------
    ndarray
        The encoded data matrix (same object as *X_encoded*).
    """
    n_cols = len(columns)
    n_rows = X_encoded.shape[0]

    for col_idx in prange(n_cols):
        col = columns[col_idx]
        X_train_col_np = train_col_values[col_idx]
        unique_vals_x = np.unique(X_encoded[:, col])
        unique_vals_train = np.unique(X_train_col_np)
        unique_vals = np.unique(np.concatenate((unique_vals_x, unique_vals_train)))

        for val in unique_vals:
            imputed_value = kappa_impute_numba(
                val,
                X_train_col_np,
                train_target_values,
                kappa_values[col_idx],
            )
            for row_idx in range(n_rows):
                if X_encoded[row_idx, col] == val:
                    X_encoded[row_idx, col] = imputed_value

    return X_encoded


@njit(parallel=True, cache=True)
def eta_predict_chunk(X_chunk, X_train, y_train, eta):
    """Predict targets for a chunk of test samples using eta-weighted distances.

    Parameters
    ----------
    X_chunk : ndarray, shape (n_test, n_features)
        Test samples.
    X_train : ndarray, shape (n_train, n_features)
        Training samples (encoded).
    y_train : ndarray, shape (n_train,)
        Training targets.
    eta : float
        Distance-decay exponent.

    Returns
    -------
    ndarray, shape (n_test,)
        Predicted values.
    """
    n_samples = X_chunk.shape[0]
    n_train = X_train.shape[0]
    n_features = X_train.shape[1]
    predictions = np.empty(n_samples, dtype=np.float32)

    for i in prange(n_samples):
        distances = np.empty(n_train, dtype=np.float32)
        for j in range(n_train):
            dist_sq = 0.0
            for k in range(n_features):
                diff = X_chunk[i, k] - X_train[j, k]
                dist_sq += diff * diff
            distances[j] = np.sqrt(dist_sq)

        weights = np.float32(1.0) / ((np.float32(1.0) + distances) ** eta)
        numerator = np.sum(y_train * weights)
        denominator = np.sum(weights)

        if denominator == 0:
            predictions[i] = np.mean(y_train)
        else:
            predictions[i] = numerator / denominator

    return predictions
