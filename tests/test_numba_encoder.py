"""Correctness of kappaeta._numba.encode_columns against a naive reference.

encode_columns memoises the encoding per unique feature value and maps rows back
by binary search. This checks that shortcut against a deliberately naive per-row
implementation, in both the near-continuous regime (almost every value unique)
and the repeated-value regime.

Runs on all platforms: numba serves the "cpu" backend everywhere, while MLX is
Apple Silicon only.
"""

import numpy as np
import pytest

from kappaeta._numba import encode_columns, kappa_impute_numba


def _naive_encode_reference(X, train_col_values, train_target_values, kappa):
    """Slow, deliberately-naive per-row reference: impute every row directly
    via kappa_impute_numba, with no unique-value memoisation at all.
    """
    n_rows, n_cols = X.shape
    out = np.empty_like(X)
    for col in range(n_cols):
        for row in range(n_rows):
            out[row, col] = kappa_impute_numba(
                X[row, col],
                train_col_values[col],
                train_target_values,
                kappa[col],
            )
    return out


@pytest.mark.parametrize("kappa_value", [0.5, 1.0, 2.5])
def test_encode_columns_matches_naive_reference_near_continuous(kappa_value):
    rng = np.random.RandomState(0)
    n_train = 200
    n_test = 300
    n_cols = 3

    # Near-continuous columns: almost every value is unique, so the memoisation
    # buys the least here and the two implementations must still agree.
    X_train = rng.uniform(0, 10, size=(n_train, n_cols)).astype(np.float32)
    y_train = rng.uniform(-5, 5, size=n_train).astype(np.float32)
    X_test = rng.uniform(0, 10, size=(n_test, n_cols)).astype(np.float32)

    columns = np.arange(n_cols)
    train_col_values = np.array([X_train[:, c] for c in columns])
    kappa_values = np.full(n_cols, kappa_value, dtype=np.float64)

    actual = encode_columns(
        X_test.copy(), columns, train_col_values, y_train, kappa_values
    )
    expected = _naive_encode_reference(
        X_test, train_col_values, y_train, kappa_values
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_encode_columns_handles_repeated_values():
    # A column with heavy repetition (categorical-like) exercises the
    # original fast path (small n_unique) and must still be correct.
    rng = np.random.RandomState(1)
    n_train = 150
    n_test = 150

    X_train_col = rng.choice([0.0, 1.0, 2.0, 3.0], size=n_train).astype(np.float32)
    y_train = rng.uniform(-1, 1, size=n_train).astype(np.float32)
    X_test_col = rng.choice([0.0, 1.0, 2.0, 3.0, 4.0], size=n_test).astype(np.float32)

    columns = np.array([0])
    train_col_values = np.array([X_train_col])
    kappa_values = np.array([1.5])

    X_test = X_test_col.reshape(-1, 1)
    actual = encode_columns(
        X_test.copy(), columns, train_col_values, y_train, kappa_values
    )
    expected = _naive_encode_reference(
        X_test, train_col_values, y_train, kappa_values
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
