"""Shared fixtures for the CPU/MLX backend parity suite.

Every fixture here mirrors KappaEtaRegressor.fit's own pipeline shape
(MinMaxScaler fit on train, applied to both splits) so parity tests exercise
realistic scaled inputs, not raw feature ranges.
"""

import numpy as np
import pytest
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import datasets as project_datasets

pytest.importorskip("mlx.core")


def _split_and_scale(X, y, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train.astype(np.float32), y_test.astype(np.float32)


# Non-network generators only -- the UCI fetchers in datasets.py require
# network access and are covered separately by the @pytest.mark.network
# tests, which are deselected by default (see pyproject.toml's addopts).
_SMALL_DATASET_GENERATORS = {
    "regression": lambda: project_datasets.generate_regression(n_samples=500),
    "polynomial": lambda: project_datasets.generate_polynomial(n_samples=500),
    "friedman1": lambda: project_datasets.generate_friedman1(n_samples=500),
    "friedman2": lambda: project_datasets.generate_friedman2(n_samples=500),
    "moons": lambda: project_datasets.generate_moons(n_samples=500),
}

_NETWORK_DATASET_NAMES = [
    "UCI: Energy Efficiency",
    "UCI: Wine",
    "UCI: Heart Failure",
    "Forest Fires",
    "Concrete",
]


@pytest.fixture(params=sorted(_SMALL_DATASET_GENERATORS))
def small_dataset(request):
    """(X_train, X_test, y_train, y_test), ~500 rows, scaled to [0, 1]."""
    X, y = _SMALL_DATASET_GENERATORS[request.param]()
    return _split_and_scale(X, y)


@pytest.fixture
def large_dataset():
    """(X_train, X_test, y_train, y_test), 15,000 rows, scaled to [0, 1].

    Large enough to force the MLX chunked path (default block size caps at
    4096) -- used for encode/predict parity only, since that path is the one
    that's genuinely memory-bounded by chunking (see kappaeta/mlx_ops.py's
    module docstring for the gradient-path memory limitation).
    """
    X, y = make_friedman1(n_samples=15000, n_features=8, noise=10.0, random_state=0)
    return _split_and_scale(X.astype(np.float32), y.astype(np.float32))


@pytest.fixture(params=_NETWORK_DATASET_NAMES)
def network_dataset(request):
    """(X_train, X_test, y_train, y_test) from a real UCI dataset (network access)."""
    X, y = project_datasets.DATASETS[request.param]()
    return _split_and_scale(X, y)
