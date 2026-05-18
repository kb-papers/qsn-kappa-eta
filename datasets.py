import numpy as np
import pandas as pd
from typing import Callable, Any
from sklearn.datasets import (
    make_friedman1,
    make_friedman2,
    make_moons,
    make_regression,
)
from ucimlrepo import fetch_ucirepo


def generate_regression(
    n_samples: int = 300,
    n_features: int = 1,
    noise_level: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple linear regression dataset."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise_level,
        random_state=random_state,
    )
    return X.astype(np.float32), y.astype(np.float32)


def generate_moons(
    n_samples: int = 300, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-circle (moons) classification dataset."""
    X, y = make_moons(noise=0.1, random_state=random_state, n_samples=n_samples)
    return X.astype(np.float32), y.astype(np.float32)


def generate_friedman1(
    n_samples: int = 300,
    n_features: int = 5,
    noise_level: float = 1.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Friedman #1: :math:`y = 10\sin(\pi x_1 x_2) + 20(x_3 - 0.5)^2 + 10 x_4 + 5 x_5 + \varepsilon`."""
    X, y = make_friedman1(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise_level,
        random_state=random_state,
    )
    return X.astype(np.float32), y.astype(np.float32)


def generate_friedman2(
    n_samples: int = 300, noise_level: float = 10.0, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_friedman2(
        n_samples=n_samples, noise=noise_level, random_state=random_state
    )
    return X.astype(np.float32), y.astype(np.float32)


def generate_polynomial(
    n_samples: int = 300,
    n_features: int = 1,
    noise_level: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Non-linear polynomial: :math:`y = x^3 - 2x^2 + x + 20\cos(x) + \varepsilon`."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-3, 3, size=(n_samples, n_features))
    x = X[:, 0]
    y = x**3 - 2 * x**2 + x + 20 * np.cos(x) + rng.normal(0, noise_level, n_samples)
    return X.astype(np.float32), y.astype(np.float32)



def _coerce_targets(y_obj: Any) -> np.ndarray:
    """Convert UCI target data to a 1-D float32 numpy array."""
    if isinstance(y_obj, pd.DataFrame):
        y = y_obj.iloc[:, 0].values if y_obj.shape[1] >= 1 else y_obj.values.ravel()
    elif isinstance(y_obj, pd.Series):
        y = y_obj.values
    else:
        y = np.asarray(y_obj)
    return y.astype(np.float32)


def _coerce_features(feat_obj: Any) -> np.ndarray:
    """Convert UCI feature data to a float32 numpy array."""
    if isinstance(feat_obj, pd.DataFrame):
        return feat_obj.values.astype(np.float32)
    return np.asarray(feat_obj).astype(np.float32)


def generate_energy_efficiency(
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """UCI Energy Efficiency dataset (768 samples, 8 features)."""
    data = fetch_ucirepo(id=242)
    return _coerce_features(data.data.features), _coerce_targets(data.data.targets)


def generate_wine(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """UCI Wine dataset (178 samples, 13 features)."""
    data = fetch_ucirepo(id=109)
    return _coerce_features(data.data.features), _coerce_targets(data.data.targets)


def generate_heart_failure(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """UCI Heart Failure Clinical Records (299 samples, 12 features)."""
    data = fetch_ucirepo(id=519)
    return _coerce_features(data.data.features), _coerce_targets(data.data.targets)


def generate_auto_mpg(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """UCI Auto MPG dataset (~392 samples, 7 features after dropping NaNs)."""
    data = fetch_ucirepo(id=9)
    X = data.data.features.values
    y = data.data.targets.values.ravel()
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    return X[mask].astype(np.float32), y[mask].astype(np.float32)


def generate_concrete(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """UCI Concrete Compressive Strength dataset (1030 samples, 8 features)."""
    data = fetch_ucirepo(id=165)
    X = data.data.features.values.astype(np.float32)
    y = data.data.targets.values.ravel().astype(np.float32)
    return X, y


def generate_forest_fires(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """UCI Forest Fires dataset (517 samples, features after one-hot encoding)."""
    data = fetch_ucirepo(id=162)
    X_df = pd.get_dummies(data.data.features, drop_first=True)
    X = X_df.values.astype(float)
    y = data.data.targets.values.ravel()
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    return X[mask].astype(np.float32), y[mask].astype(np.float32)


N_SAMPLES = 1000

DATASETS: dict[str, Callable[[], tuple[np.ndarray, np.ndarray]]] = {
    "Friedman2": lambda: generate_friedman2(n_samples=N_SAMPLES, noise_level=10.0),
    "Friedman1": lambda: generate_friedman1(n_samples=N_SAMPLES, noise_level=10.0),
    "Moons": lambda: generate_moons(n_samples=N_SAMPLES),
    "Regression": lambda: generate_regression(n_samples=N_SAMPLES, noise_level=0.1),
    "Polynomial Regression": lambda: generate_polynomial(n_samples=N_SAMPLES),
    "UCI: Energy Efficiency": generate_energy_efficiency,
    "UCI: Wine": generate_wine,
    "UCI: Heart Failure": generate_heart_failure,
    "Forest Fires": generate_forest_fires,
    "Concrete": generate_concrete,
}

INIT_STRATEGIES: dict[str, tuple[float, float]] = {
    "01": (1.0, 1.0),
    "02": (5.0, 5.0),
    "03": (10.0, 10.0),
    "04": (20.0, 20.0),
    "05": (30.0, 30.0),
    "06": (50.0, 50.0),
    "07": (60.0, 60.0),
    "08": (70.0, 70.0),
}
