"""The 20 benchmark datasets and the tuning grids of the paper's Methodology.

``DATASETS`` maps each display name to its loader; the five synthetic sets come
from scikit-learn and the fifteen real ones from the UCI repository, fetched via
``ucimlrepo``. ``INIT_STRATEGIES`` and ``OPT_GRID`` are the five initializations
and three per-optimizer configurations every gradient method is given.
"""

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
    """Convert UCI target data to a 1-D float32 numpy array.

    String class labels (e.g. "M"/"B") are factorised to integer codes first --
    plain astype(float32) would raise on non-numeric dtypes.
    """
    if isinstance(y_obj, pd.DataFrame):
        y = y_obj.iloc[:, 0].values if y_obj.shape[1] >= 1 else y_obj.values.ravel()
    elif isinstance(y_obj, pd.Series):
        y = y_obj.values
    else:
        y = np.asarray(y_obj)
    if y.dtype.kind not in "iuf":
        y = pd.factorize(y)[0]
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


def generate_real_estate_valuation(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """UCI Real Estate Valuation dataset (414 samples, 6 features)."""
    data = fetch_ucirepo(id=477)
    return _coerce_features(data.data.features), _coerce_targets(data.data.targets)


def generate_airfoil_self_noise(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """UCI Airfoil Self-Noise dataset (1503 samples, 5 features)."""
    data = fetch_ucirepo(id=291)
    return _coerce_features(data.data.features), _coerce_targets(data.data.targets)


def generate_ionosphere(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """UCI Ionosphere dataset (351 samples, 34 features, binary class label as target)."""
    data = fetch_ucirepo(id=52)
    return _coerce_features(data.data.features), _coerce_targets(data.data.targets)


def generate_sonar(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """UCI Connectionist Bench (Sonar, Mines vs. Rocks) dataset (208 samples, 60 features, binary class label as target)."""
    data = fetch_ucirepo(id=151)
    return _coerce_features(data.data.features), _coerce_targets(data.data.targets)


def generate_glass_identification(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """UCI Glass Identification dataset (214 samples, 9 features, glass-type label as target)."""
    data = fetch_ucirepo(id=42)
    return _coerce_features(data.data.features), _coerce_targets(data.data.targets)


def generate_statlog_heart(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """UCI Statlog (Heart) dataset (270 samples, 13 features, presence-of-disease label as target)."""
    data = fetch_ucirepo(id=145)
    return _coerce_features(data.data.features), _coerce_targets(data.data.targets)


def generate_blood_transfusion(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """UCI Blood Transfusion Service Center dataset (748 samples, 4 features, donation label as target)."""
    data = fetch_ucirepo(id=176)
    return _coerce_features(data.data.features), _coerce_targets(data.data.targets)


def generate_parkinsons(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """UCI Parkinsons dataset (195 samples, 22 features, status label as target)."""
    data = fetch_ucirepo(id=174)
    return _coerce_features(data.data.features), _coerce_targets(data.data.targets)


def generate_wine_quality(
    random_state: int = 42, n_samples: int = 1600
) -> tuple[np.ndarray, np.ndarray]:
    """UCI Wine Quality dataset (combined red+white, id=186), subsampled to n_samples.

    The full set is 6497 rows, far larger than every other dataset here. The
    ``DATASETS`` entry passes ``n_samples=100_000`` to load it in full via the
    ``min(n_samples, len(X))`` clamp below; the panel run compensates for its size
    with a 256-candidate grid resolution (``SEARCH_BUDGET_OVERRIDES`` in
    ``02_optimizer_comparison_run.ipynb``) rather than by subsampling rows.
    """
    data = fetch_ucirepo(id=186)
    X = _coerce_features(data.data.features)
    y = _coerce_targets(data.data.targets)
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    return X[idx], y[idx]


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

    # The panel is 20 dataset-blocks because the Friedman omnibus test gains its
    # power from the number of blocks, not from the number of runs within one.
    # These ten are all real UCI sets, kept modestly sized (~200-1600 rows) so the
    # full run stays tractable.
    "UCI: Auto MPG": generate_auto_mpg,
    "UCI: Real Estate Valuation": generate_real_estate_valuation,
    "UCI: Airfoil Self-Noise": generate_airfoil_self_noise,
    "UCI: Ionosphere": generate_ionosphere,
    "UCI: Sonar": generate_sonar,
    "UCI: Wine Quality": lambda: generate_wine_quality(n_samples=100_000),
    "UCI: Glass Identification": generate_glass_identification,
    "UCI: Statlog Heart": generate_statlog_heart,
    "UCI: Blood Transfusion": generate_blood_transfusion,
    "UCI: Parkinsons": generate_parkinsons,
}


INIT_STRATEGIES: dict[str, tuple[float, float]] = {
    "01": (0.0, 0.0),
    "02": (20.0, 20.0),
    "03": (40.0, 40.0),
    "04": (60.0, 60.0),
    "05": (80.0, 80.0),
}

N_CFGS_PER_OPTIMIZER = 3

OPT_GRID: dict[str, list[dict]] = {

    "Adam": [
        {"learning_rate": lr}
        for lr in (0.1, 0.5, 1.0)
    ],

    # initial_step_size is our L-BFGS's live knob, over the same three values as
    # Adam's learning rate. memory_size is held fixed rather than tuned, so that
    # the varied axis is the one the two first- and second-order methods have in
    # common and each method still gets exactly three configurations.
    "L-BFGS": [
        {"initial_step_size": a, "memory_size": 5}
        for a in (0.1, 0.5, 1.0,)
    ],

    # scipy's L-BFGS-B picks its own step via a Wolfe line search, so it exposes no
    # initial-step-size analogue. maxcor, the number of stored correction pairs, is
    # the axis varied instead, and maxls is left at scipy's default of 20.
    "L-BFGS-B": [{"maxcor": m} for m in (3, 5, 7)],
}

assert {len(v) for v in OPT_GRID.values()} == {N_CFGS_PER_OPTIMIZER}, (
    "OPT_GRID must give every optimizer an equal number of candidate configurations"
)
