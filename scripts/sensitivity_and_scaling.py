"""The epsilon sensitivity sweep and the scaling study of the paper's appendix.

1. ``epsilon_sweep`` - the distance-smoothing floor epsilon across the range
   1e-16..1e-4, reporting the gradients at three parameter regimes and the
   optimum each choice produces. This is demonstrates that the shipped
   value of 1e-12 is not load-bearing.
2. ``scaling_study``  - gradient and Hessian time against the reference-set size
   N at several (M, d), with an OLS log-log slope and a 95% confidence interval
   per series, plus peak memory. This is what tests the claim that the Hessian
   preserves the gradient's order in N rather than raising it.

The gradient path is MLX, the backend the optimizer experiments run on; the
Hessian path is JAX ``jacfwd(jacrev(.))``, because MLX exposes no second-order
transform.

Writes the underlying measurements to results/.

Run:  .venv/bin/python scripts/sensitivity_and_scaling.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import methodology as meth  # noqa: E402
from kappaeta import MLX_AVAILABLE  # noqa: E402

if MLX_AVAILABLE:
    import mlx.core as mx

    from kappaeta import mlx_ops

RESULTS = ROOT / "results"

EPS_DECADES = [10.0 ** -k for k in range(16, 3, -1)]   # 1e-16 ... 1e-4
EPS_REGIMES = [(1.0, 1.0), (5.0, 5.0), (10.0, 10.0)]
SHIPPED_EPS = 1e-12


# --- an epsilon-parameterised copy of the shipped kernel --------------------------

def _encode_col(x_query, x_train_col, y_train, kappa):
    distances = jnp.abs(x_query[:, None] - x_train_col[None, :])
    log_w = -kappa * jnp.log1p(distances)
    w = jnp.exp(log_w - jnp.max(log_w, axis=1, keepdims=True))
    return jnp.sum(y_train[None, :] * w, axis=1) / (jnp.sum(w, axis=1) + 1e-12)


def _loss_eps(params, X_eval, X_train, y_train, y_eval, eps):
    kappa = jnp.atleast_1d(params["kappa"])
    cols_eval = [_encode_col(X_eval[:, p], X_train[:, p], y_train, kappa[p])
                 for p in range(X_train.shape[1])]
    cols_train = [_encode_col(X_train[:, p], X_train[:, p], y_train, kappa[p])
                  for p in range(X_train.shape[1])]
    Xe, Xt = jnp.stack(cols_eval, axis=1), jnp.stack(cols_train, axis=1)

    diff = Xe[:, None, :] - Xt[None, :, :]
    distances = jnp.sqrt(jnp.sum(diff ** 2, axis=2) + eps)
    log_w = -params["eta"] * jnp.log1p(distances)
    w = jnp.exp(log_w - jnp.max(log_w, axis=1, keepdims=True))
    y_pred = jnp.sum(y_train[None, :] * w, axis=1) / (jnp.sum(w, axis=1) + 1e-12)
    return jnp.mean((y_eval - y_pred) ** 2)


_grad_eps = jax.jit(jax.grad(_loss_eps), static_argnums=())


def _polynomial_split(seed=42):
    """The univariate polynomial set of the paper's gradient validation, under the
    same three-way split the rest of the protocol uses."""
    from datasets import generate_polynomial

    X, y = generate_polynomial(n_samples=400, random_state=seed)
    return meth.three_way_split(X, y, seed)


# --- 1. epsilon sweep -------------------------------------------------------------

def epsilon_sweep() -> pd.DataFrame:
    X_train, y_train, X_val, y_val, _, _ = _polynomial_split()
    X_train, y_train = jnp.array(X_train), jnp.array(y_train)
    X_val, y_val = jnp.array(X_val), jnp.array(y_val)
    n_features = X_train.shape[1]

    rows = []
    for eps in EPS_DECADES:
        for kappa0, eta0 in EPS_REGIMES:
            params = {"kappa": jnp.full(n_features, kappa0), "eta": jnp.array(eta0)}
            g = _grad_eps(params, X_val, X_train, y_train, y_val, eps)
            loss = float(_loss_eps(params, X_val, X_train, y_train, y_val, eps))
            rows.append(dict(eps=eps, kappa0=kappa0, eta0=eta0, loss=loss,
                             grad_kappa=float(jnp.mean(g["kappa"])), grad_eta=float(g["eta"])))

        def fun_and_jac(x, eps=eps):
            p = {"kappa": jnp.array(x[:n_features]), "eta": jnp.array(x[n_features])}
            val = _loss_eps(p, X_val, X_train, y_train, y_val, eps)
            g = _grad_eps(p, X_val, X_train, y_train, y_val, eps)
            return float(val), np.concatenate([np.asarray(g["kappa"], dtype=np.float64),
                                               [float(g["eta"])]])

        best = min(
            (minimize(fun_and_jac, np.full(n_features + 1, float(c)), jac=True,
                      method="L-BFGS-B", bounds=[(1.0, 80.0)] * (n_features + 1),
                      options={"maxiter": 200})
             for c in (1.0, 20.0, 40.0, 60.0, 80.0)),
            key=lambda r: r.fun,
        )
        for row in rows[-len(EPS_REGIMES):]:
            row.update(opt_kappa=float(np.mean(best.x[:n_features])),
                       opt_eta=float(best.x[n_features]), opt_loss=float(best.fun))
    return pd.DataFrame(rows)


# --- 2. scaling study -------------------------------------------------------------

def _make_arrays(n_train, n_eval, n_features, seed=0):
    rng = np.random.RandomState(seed)
    return (rng.uniform(0, 1, (n_train, n_features)).astype(np.float32),
            rng.uniform(0, 1, (n_eval, n_features)).astype(np.float32),
            rng.uniform(-1, 1, n_train).astype(np.float32),
            rng.uniform(-1, 1, n_eval).astype(np.float32))


def _median_time(fn, repeats):
    fn()  # warm the compilation cache before the timed region
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000.0


_hessian_fn = jax.jit(jax.jacfwd(jax.jacrev(
    lambda params, X_eval, X_train, y_train, y_eval:
        _loss_eps(params, X_eval, X_train, y_train, y_eval, SHIPPED_EPS)
)))


def _time_gradient(X_train, X_eval, y_train, y_eval, kappa0=2.0, eta0=2.0, repeats=15):
    def run():
        gk, ge = mlx_ops.compute_gradients_mlx(kappa0, eta0, X_eval, X_train, y_train, y_eval)
        mx.eval(gk, ge)
    mx.reset_peak_memory()
    ms = _median_time(run, repeats)
    return ms, mx.get_peak_memory() / 1e6


def _time_hessian(X_train, X_eval, y_train, y_eval, kappa0=2.0, eta0=2.0, repeats=5):
    Xt, Xe = jnp.array(X_train), jnp.array(X_eval)
    yt, ye = jnp.array(y_train), jnp.array(y_eval)
    params = {"kappa": jnp.full(X_train.shape[1], kappa0), "eta": jnp.array(eta0)}

    def run():
        H = _hessian_fn(params, Xe, Xt, yt, ye)
        jax.block_until_ready(H)
    return _median_time(run, repeats)


def fit_slope(n_values, times_ms):
    """OLS slope of log(time) on log(N), with a 95% CI from the residual standard
    error. The slope is the empirical complexity exponent; the interval is what
    makes it a measurement rather than a visual impression."""
    x, y = np.log(np.asarray(n_values, float)), np.log(np.asarray(times_ms, float))
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    se = np.sqrt((resid ** 2).sum() / (n - 2) / ((x - x.mean()) ** 2).sum())
    half = student_t.ppf(0.975, n - 2) * se
    return slope, slope - half, slope + half


ALL_N = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]

CONFIG_MAX_N = {(10, 3): 6000, (30, 3): 3000, (100, 3): 2000, (10, 8): 3000, (10, 20): 2000}


def scaling_study(quick=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_values = ALL_N
    configs = list(CONFIG_MAX_N)
    grad_reps, hess_reps = (5, 3) if quick else (15, 7)
    if quick:
        n_values = n_values[::3]
        configs = configs[:2]

    rows = []
    for M, d in configs:
        for N in [n for n in n_values if n <= CONFIG_MAX_N[(M, d)]]:
            X_train, X_eval, y_train, y_eval = _make_arrays(N, M, d)
            g_ms, g_mem = _time_gradient(X_train, X_eval, y_train, y_eval, repeats=grad_reps)
            h_ms = _time_hessian(X_train, X_eval, y_train, y_eval, repeats=hess_reps)
            rows.append(dict(M=M, d=d, N=N, grad_ms=g_ms, grad_peak_mb=g_mem, hess_ms=h_ms))
            print(f"  M={M:3d} d={d:2d} N={N:5d}  grad {g_ms:8.2f} ms ({g_mem:7.1f} MB)  "
                  f"hess {h_ms:8.2f} ms")
    timings = pd.DataFrame(rows)

    # Two fits per series. The full-range fit is biased downwards because at small
    # N a call's fixed dispatch and kernel-launch overhead is a large share of its
    # cost; the asymptotic fit drops the smallest third of the sizes, where that
    # floor dominates, and is the one that estimates the exponent the analysis
    # predicts. Both are reported rather than only the flattering one.
    slopes = []
    for (M, d), grp in timings.groupby(["M", "d"]):
        sizes = sorted(grp["N"].unique())
        asymptotic_from = sizes[len(sizes) // 3]
        for kind, col in (("Gradient", "grad_ms"), ("Hessian", "hess_ms")):
            for label, sub in (("all", grp), (f"$N \\geq {asymptotic_from}$",
                                              grp[grp.N >= asymptotic_from])):
                s, lo, hi = fit_slope(sub["N"], sub[col])
                slopes.append(dict(M=M, d=d, quantity=kind, fit_range=label,
                                   slope=s, lo=lo, hi=hi))
    return timings, pd.DataFrame(slopes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer sizes and repetitions")
    args = ap.parse_args()

    print("=== epsilon sweep ===")
    sweep = epsilon_sweep()
    sweep.to_csv(RESULTS / "sensitivity-epsilon-sweep.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(sweep.to_string(index=False))

    print("\n=== scaling study ===")
    timings, slopes = scaling_study(quick=args.quick)
    timings.to_csv(RESULTS / "scaling-timings.csv", index=False)
    slopes.to_csv(RESULTS / "scaling-slopes.csv", index=False)
    print("\nfitted log-log slopes:")
    print(slopes.to_string(index=False))


if __name__ == "__main__":
    main()
