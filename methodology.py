"""The experiment protocol of the paper's Methodology and Results sections.

Holds the pieces shared by the run notebooks and by ``03_results_and_significance.ipynb``:
the 60/20/20 split, the leakage-guarded single test read, one runner per optimizer,
the t_max-budgeted (configuration x initialisation) search, the column-incremental
grid-search baseline, the definition of the 20-dataset panel, and the
rank/Friedman/Wilcoxon-Holm procedure the paper follows after Demsar (2006).

Every function takes a ``backend`` argument resolved by ``kappaeta._backend``, so
the protocol runs on MLX where it is available and on the CPU path elsewhere.
"""

import os
import time
from decimal import Decimal
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from kappaeta import AdamOptimizer, LBFGSOptimizer
from kappaeta._backend import resolve_backend


def _resolve_ops(backend=None):
    """Return (xp, pipeline_forward, mse_loss, loss_function, compute_loss_and_grad)
    for the resolved backend, mirroring the dispatch
    ``kappaeta.optimizers._BaseOptimizer`` does internally: MLX under
    ``backend="mlx"``, JAX under ``backend="cpu"``.
    """
    resolved = resolve_backend(backend)
    if resolved == "mlx":
        import mlx.core as xp

        from kappaeta import mlx_ops as ops

        return (
            xp, ops.pipeline_forward_mlx, ops.mse_loss_mlx,
            ops.loss_function_mlx, ops.compute_loss_and_grad_mlx,
        )
    import jax.numpy as xp

    from kappaeta import jax_ops as ops

    return (
        xp, ops.pipeline_forward_jax, ops.mse_loss,
        ops.loss_function_jax, ops.compute_loss_and_grad_jax,
    )



# --- The 20-dataset panel ---------------------------------------------------------
# The panel the paper reports is the 20-dataset run, Wine Quality included at its
# full 6497 rows. Wine Quality is the only dataset run at a 256-candidate grid
# resolution rather than 2048, because grid search's measured cost sets every
# group's t_max and scales with N^2. This is sound because t_max is measured
# *within* each (dataset, seed) group, so the gradient methods there are budgeted
# against exactly the baseline they are compared against, and across datasets
# only ranks are ever compared.
#
# This is the single definition of the panel: 02_optimizer_comparison_run.ipynb
# produces it and 03_results_and_significance.ipynb reads it from here, so no
# analysis can silently run on a different set of rows than the paper reports.

ROOT = Path(__file__).resolve().parent
PANEL_CSV = ROOT / "results/results-tmax-gridmatched-perfeature-cfg3-init5-ds20-grid2048.csv"

ALGORITHMS = ["Adam", "L-BFGS", "L-BFGS-B", "Grid search"]
GRADIENT_ALGORITHMS = ["Adam", "L-BFGS", "L-BFGS-B"]


def load_panel():
    """Return every run of the paper's 20-dataset panel as one long DataFrame."""
    df = pd.read_csv(PANEL_CSV)
    n_datasets = df["dataset"].nunique()
    assert n_datasets == 20, f"panel must hold 20 datasets, found {n_datasets}"
    assert set(df["optimizer"]) == set(ALGORITHMS)
    return df


def per_group_accuracy(df):
    """One test MSE per (dataset, optimizer, seed): the single row each group is
    allowed to read test on, at its own validation argmin."""
    acc = df[df["test_mse"].notna()]
    counts = acc.groupby(["dataset", "optimizer", "seed"]).size()
    assert (counts == 1).all(), "each (dataset, optimizer, seed) group must read test exactly once"
    assert len(acc) == 800, f"expected 800 test reads across the panel, found {len(acc)}"
    return acc[["dataset", "optimizer", "seed", "test_mse"]]


def per_group_cost(df):
    """Total tuning wall-clock per (dataset, optimizer, seed): summed over every
    (cfg, init) run the method spent its budget on, not just the winning one."""
    return df.groupby(["dataset", "optimizer", "seed"])["wall_clock"].sum().reset_index()


def wide_mean(long_df, value):
    """(dataset x optimizer) table of the per-group mean across the ten seeds."""
    return long_df.groupby(["dataset", "optimizer"])[value].mean().unstack("optimizer")[ALGORITHMS]


def wide_sd(long_df, value):
    """(dataset x optimizer) table of the per-group standard deviation across seeds."""
    return long_df.groupby(["dataset", "optimizer"])[value].std().unstack("optimizer")[ALGORITHMS]


# --- Data splitting -----------------------------------------------------------

def three_way_split(X, y, seed):
    """60/20/20 train/val/test. Scaler is fit on train only."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return (
        X_train, y_train.astype(np.float32),
        X_val, y_val.astype(np.float32),
        X_test, y_test.astype(np.float32),
    )


# --- Leakage-guarded test evaluation -------------------------------------------

def evaluate_test_once(all_test_reads, label, X_test, X_train, y_train, y_test, kappa, eta, backend=None):
    """Forward-evaluate test MSE once.

    Callers must route every test-set read through this function and own the
    ``all_test_reads`` list they pass in, so a post-hoc assertion can confirm
    no group read test more than once.
    """
    xp, pipeline_forward, mse_loss_fn, _, _ = _resolve_ops(backend)
    all_test_reads.append(label)
    y_pred = pipeline_forward(X_test, X_train, y_train, kappa, eta)
    return float(mse_loss_fn(xp.array(y_test), y_pred))


def _final_val_mse(kappa, eta, X_train, y_train, X_val, y_val, backend=None):
    """Recompute val MSE at the optimizer's *returned* (kappa, eta).

    Not read from its history: each optimizer's last history entry is recorded
    before that iteration's parameter update, so it can lag the returned
    params by one step when the loop runs to max_iters/time_budget without
    early convergence.
    """
    xp, _, _, loss_function, _ = _resolve_ops(backend)
    params = {
        "kappa": xp.array(kappa, dtype=xp.float32),
        "eta": xp.array(eta, dtype=xp.float32),
    }
    loss = loss_function(
        params, xp.array(X_val), xp.array(X_train), xp.array(y_train), xp.array(y_val)
    )
    return float(loss)


# --- Gradient-based optimizer runners ------------------------------------------
# All three take the same (cfg, kappa0, eta0, X_train, y_train, X_val, y_val)
# signature plus shared keyword-only run parameters, and return the same result
# dict shape, so a caller can treat them uniformly. The validation split is what
# every method minimises; the test split is read once per group, afterwards, by
# evaluate_test_once.

def run_adam(cfg, kappa0, eta0, X_train, y_train, X_val, y_val, *,
             max_iters, tol, bounds, time_budget, backend=None):
    opt = AdamOptimizer(
        max_iters=max_iters, tol=tol, time_budget=time_budget,
        kappa_bounds=bounds, eta_bounds=bounds, verbose=False, backend=backend, **cfg,
    )
    t0 = time.time()
    kappa, eta, _history = opt.optimize(kappa0, eta0, X_val, X_train, y_train, y_val)
    wall = time.time() - t0
    n_kappa = np.atleast_1d(kappa0).shape[0]
    assert np.atleast_1d(kappa).shape[0] == n_kappa, (
        f"Adam returned a {np.atleast_1d(kappa).shape[0]}-dim kappa for a {n_kappa}-dim kappa0"
    )
    val_mse = _final_val_mse(kappa, eta, X_train, y_train, X_val, y_val, backend=backend)
    return dict(
        kappa=kappa, eta=eta, val_mse=val_mse, wall_clock=wall,
        nfev=opt.nfev, njev=opt.njev, n_linesearch=opt.n_linesearch, nit=opt.nit,
        stop_reason=opt.stop_reason,
    )


def run_lbfgs(cfg, kappa0, eta0, X_train, y_train, X_val, y_val, *,
              max_iters, tol, bounds, time_budget, backend=None):
    opt = LBFGSOptimizer(
        max_iters=max_iters, tol=tol, time_budget=time_budget,
        kappa_bounds=bounds, eta_bounds=bounds, verbose=False, backend=backend, **cfg,
    )
    t0 = time.time()
    kappa, eta, _history = opt.optimize(kappa0, eta0, X_val, X_train, y_train, y_val)
    wall = time.time() - t0
    n_kappa = np.atleast_1d(kappa0).shape[0]
    assert np.atleast_1d(kappa).shape[0] == n_kappa, (
        f"L-BFGS returned a {np.atleast_1d(kappa).shape[0]}-dim kappa for a {n_kappa}-dim kappa0"
    )
    val_mse = _final_val_mse(kappa, eta, X_train, y_train, X_val, y_val, backend=backend)
    return dict(
        kappa=kappa, eta=eta, val_mse=val_mse, wall_clock=wall,
        nfev=opt.nfev, njev=opt.njev, n_linesearch=opt.n_linesearch, nit=opt.nit,
        stop_reason=opt.stop_reason,
    )


def run_lbfgsb(cfg, kappa0, eta0, X_train, y_train, X_val, y_val, *,
               max_iters, tol, bounds, time_budget, backend=None):
    """scipy L-BFGS-B: box-constrained, nfev/njev/nit come for free from OptimizeResult.

    scipy has no native time-limit option, so ``time_budget`` is enforced with
    a callback that raises StopIteration (scipy's ``_call_callback_maybe_halt``
    catches it and halts the run). The callback path needs its own
    ``halted_on_time`` flag rather than reading ``result.status``: a callback
    halt leaves ``warnflag == 2`` ("abnormal") since neither the maxfun nor
    maxiter branch fired, and the callback-halt assignment is followed by a
    non-elif maxiter check that can overwrite the message.
    """
    xp, _, _, _, compute_loss_and_grad = _resolve_ops(backend)
    X_train_a, y_train_a = xp.array(X_train), xp.array(y_train)
    X_val_a, y_val_a = xp.array(X_val), xp.array(y_val)

    kappa0_arr = np.atleast_1d(np.asarray(kappa0, dtype=np.float64))
    n_kappa = kappa0_arr.shape[0]

    def fun_and_jac(x):
        kappa = xp.array(x[:n_kappa], dtype=xp.float32)
        eta = xp.array(x[n_kappa], dtype=xp.float32)
        loss, grad_kappa, grad_eta = compute_loss_and_grad(
            kappa, eta, X_val_a, X_train_a, y_train_a, y_val_a
        )
        grad_kappa_flat = np.atleast_1d(np.array(grad_kappa, dtype=np.float64))
        grad = np.concatenate([grad_kappa_flat, [float(grad_eta)]])
        return float(loss), grad

    state = {"halted_on_time": False}
    t0 = time.time()

    # Parameter must be named intermediate_result: scipy's _wrap_callback
    # switches on that name to decide whether to pass the OptimizeResult.
    def callback(intermediate_result):
        if time_budget is not None and (time.time() - t0) > time_budget:
            state["halted_on_time"] = True
            raise StopIteration

    x0 = np.concatenate([kappa0_arr, [float(eta0)]])
    result = scipy_minimize(
        fun_and_jac, x0, jac=True, method="L-BFGS-B", bounds=[bounds] * (n_kappa + 1), callback=callback,
        options={"maxiter": max_iters, "maxcor": cfg["maxcor"], "ftol": tol, "gtol": tol},
    )
    wall = time.time() - t0

    if state["halted_on_time"]:
        stop_reason = "time_budget"      # must precede the status checks -- see docstring
    elif result.status == 0:
        stop_reason = "converged"        # ftol or gtol satisfied
    elif "ITERATIONS" in result.message:
        stop_reason = "max_iters"        # task 504
    elif "EVALUATIONS" in result.message:
        stop_reason = "max_fev"          # task 502, maxfun (scipy default 15000)
    else:
        # "ABNORMAL:" with an empty task message == ABNORMAL_TERMINATION_IN_LNSRCH,
        # i.e. converged to float32 precision and the line search stalled.
        stop_reason = "abnormal"

    kappa, eta = result.x[:n_kappa], float(result.x[n_kappa])
    assert kappa.shape[0] == n_kappa, f"L-BFGS-B returned a {kappa.shape[0]}-dim kappa for a {n_kappa}-dim kappa0"
    # Recomputed rather than read off result.fun: on the callback-halt path
    # result.fun need not correspond to the returned result.x.
    val_mse = _final_val_mse(kappa, eta, X_train, y_train, X_val, y_val, backend=backend)

    return dict(
        kappa=kappa, eta=eta, val_mse=val_mse, wall_clock=wall,
        # nfev == njev is expected: with jac=True scipy wraps the callable in
        # MemoizeJac, so both counters advance once per underlying evaluation.
        # n_linesearch stays None -- scipy exposes no line-search trial count.
        nfev=int(result.nfev), njev=int(result.njev), n_linesearch=None, nit=int(result.nit),
        stop_reason=stop_reason,
    )


# --- t_max-budgeted tuning search -----------------------------------------------

def run_tuning_search(run_fn, cfg_grid, init_strategies, X_train, y_train, X_val, y_val, *,
                       t_max, max_iters, tol, bounds, n_features, backend=None):
    """Grid search over every (cfg, init) combination, with the total search
    time capped at ``t_max`` seconds by splitting it evenly across the grid:
    each combination gets ``t_max / (len(cfg_grid) * len(init_strategies))``
    seconds of wall-clock budget.

    ``init_strategies``' scalar ``(kappa0, eta0)`` starting points are
    broadcast to a per-feature kappa vector here (``np.full(n_features,
    kappa0)``), so every run below optimises a genuine ``n_features``-dim
    kappa rather than one scalar shared across all features -- the actual
    optimizers (``AdamOptimizer``/``LBFGSOptimizer``/scipy L-BFGS-B) are
    shape-generic and need no further change to do this.

    Returns
    -------
    grid_runs : list of (cfg, init_name, result_dict)
        Every combination's result, in grid order.
    best : (cfg, init_name, result_dict)
        The val_mse-argmin entry -- the one that proceeds to the test phase.
    """
    per_combo_budget = t_max / (len(cfg_grid) * len(init_strategies))
    grid_runs = [
        (cfg, init_name, run_fn(
            cfg, np.full(n_features, kappa0, dtype=np.float64), eta0, X_train, y_train, X_val, y_val,
            max_iters=max_iters, tol=tol, bounds=bounds, time_budget=per_combo_budget, backend=backend,
        ))
        for cfg in cfg_grid
        for init_name, (kappa0, eta0) in init_strategies.items()
    ]
    best_idx = min(range(len(grid_runs)), key=lambda i: grid_runs[i][2]["val_mse"])
    return grid_runs, grid_runs[best_idx]


# --- Derivative-free baseline ---------------------------------------------------

def _staircase_candidates(bounds, n_dims, budget):
    """Column-incremental walk through an ``n_dims``-D grid lattice, budget-limited.

    A full factorial grid needs ``n_levels**n_dims`` candidates -- infeasible
    once kappa is per-feature (a 60-feature dataset has 61 dims). Instead this
    starts at the low corner and advances one dimension at a time to the next
    grid level (holding every other dimension at its most recently set
    level), cycling through dimensions 0..n_dims-1 and climbing to further
    levels as long as the budget allows: e.g. for n_dims=4,
    ``[0,0,0,0] -> [40,0,0,0] -> [40,40,0,0] -> [40,40,40,0] -> [40,40,40,40]
    -> [80,40,40,40] -> ...``. Cost is ``1 + (n_levels - 1) * n_dims``
    candidates -- linear in ``n_dims``, not exponential -- using as many
    levels as fit: ``n_levels = 1 + (budget - 1) // n_dims``.
    """
    n_levels = 1 + (budget - 1) // n_dims
    if n_levels < 2:
        print(
            f"run_grid_search_baseline: budget={budget} is too small for {n_dims} dimensions "
            f"(need >= {n_dims + 1} for any exploration beyond the starting corner); "
            f"only the all-low corner will be evaluated."
        )
    grid_vals = np.linspace(bounds[0], bounds[1], n_levels)

    current = [grid_vals[0]] * n_dims
    candidates = [tuple(current)]
    for level_idx in range(1, n_levels):
        for dim in range(n_dims):
            current[dim] = grid_vals[level_idx]
            candidates.append(tuple(current))
    return candidates


def run_grid_search_baseline(X_train, y_train, X_val, y_val, bounds, budget, backend=None):
    """Column-incremental grid search over the full (kappa_1, ..., kappa_p, eta)
    space: no initialisation or optimizer-configuration concept, a matched
    evaluation budget, and validation only. This is the paper's derivative-free
    baseline, and its measured cost on a group sets that group's ``t_max``. See
    ``_staircase_candidates`` for the traversal it uses in place of a full
    factorial grid, which is infeasible once kappa is per-feature.
    """
    xp, _, _, loss_function, _ = _resolve_ops(backend)
    n_features = X_train.shape[1]
    n_dims = n_features + 1  # one kappa per feature, plus eta
    candidates = _staircase_candidates(bounds, n_dims, budget)

    X_train_a, y_train_a = xp.array(X_train), xp.array(y_train)
    X_val_a, y_val_a = xp.array(X_val), xp.array(y_val)

    def candidate_val_mse(point):
        kappa = xp.array(point[:n_features], dtype=xp.float32)
        eta = xp.array(point[n_features], dtype=xp.float32)
        params = {"kappa": kappa, "eta": eta}
        return float(loss_function(params, X_val_a, X_train_a, y_train_a, y_val_a))

    t0 = time.time()
    best_point, best_val_mse = min(
        ((point, candidate_val_mse(point)) for point in candidates),
        key=lambda t: t[1],
    )
    wall = time.time() - t0
    kappa_best = np.array(best_point[:n_features])
    eta_best = float(best_point[n_features])
    assert kappa_best.shape[0] == n_features, (
        f"Grid search baseline returned a {kappa_best.shape[0]}-dim kappa for {n_features} features"
    )
    return dict(
        kappa=kappa_best, eta=eta_best, val_mse=best_val_mse, wall_clock=wall, n_evals=len(candidates),
    )


# --- Result persistence ----------------------------------------------------------

def save_upsert(df, path, key_cols):
    """Upsert ``df``'s rows into the CSV at ``path``, keyed by ``key_cols``.

    NaN (from a re-loaded CSV) and None (from an in-memory frame) stringify
    differently, which would silently break the key match for rows with null
    key columns (e.g. baselines with no init/cfg) -- both are normalised to
    the placeholder ``"NA"`` first. Re-running with the same keys replaces
    those rows rather than duplicating them.

    Returns
    -------
    combined : pd.DataFrame
    n_replaced : int
    """
    def _row_keys(frame):
        key_frame = frame[key_cols].copy()
        key_frame[key_cols] = key_frame[key_cols].fillna("NA")
        return key_frame.astype(str).agg("|".join, axis=1)

    new_df = df.copy()
    new_keys = _row_keys(new_df)

    if os.path.exists(path):
        existing_df = pd.read_csv(path, index_col=0, dtype={"init": str})
        existing_keys = _row_keys(existing_df)
        n_replaced = int(existing_keys.isin(new_keys).sum())
        combined = pd.concat([existing_df[~existing_keys.isin(new_keys)], new_df], ignore_index=True)
    else:
        n_replaced = 0
        combined = new_df

    combined.to_csv(path)
    return combined, n_replaced


# --- Display formatting -----------------------------------------------------------

def fmt_sigfig(x, n=4):
    """Format ``x`` to ``n`` significant figures as a fixed-point or scientific string.

    Unlike ``f"{x:.4g}"`` (which strips trailing zeros, e.g. ``0.0207`` prints
    as ``"0.0207"`` -- only 3 significant figures), this pads to exactly ``n``
    significant digits (``"0.02070"``) so columns of wildly different
    magnitude (this protocol's MSEs span ~0.0002 to ~8000) are each shown at
    the same precision rather than the same decimal-place count. Falls back to
    scientific notation outside [1e-4, 1e6) (same threshold Python's ``%g``
    uses) so a tiny Friedman p-value doesn't print as a wall of leading zeros.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NaN"
    x = float(x)
    if x == 0:
        return f"{0:.{n - 1}f}"
    exponent = Decimal(repr(x)).adjusted()
    if exponent < -4 or exponent >= 6:
        return f"{x:.{n - 1}e}"
    decimals = max(n - 1 - exponent, 0)
    return f"{x:.{decimals}f}"


# --- Significance testing: aggregate + rank + Friedman + Wilcoxon/Holm -----------

def holm_bonferroni(pvals):
    """Holm step-down correction for a list of raw p-values."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, min((m - rank) * pvals[idx], 1.0))
        adjusted[idx] = running_max
    return adjusted


def all_pairs_wilcoxon(wide_df, algorithms, alpha=0.05, names=None):
    """Wilcoxon signed-rank on every unordered pair of methods, Holm-corrected.

    This is the post-hoc the paper reports: all six comparisons among the four
    methods, rather than a control-versus-rest set, so no conclusion rests on the
    control having been chosen from the same data.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Index = dataset, one column per name in ``algorithms``.
    algorithms : list of str
    alpha : float
    names : dict, optional
        Display names for the ``comparison`` column. Defaults to the raw names.
    """
    label = (lambda a: a) if names is None else (lambda a: names[a])
    rows = []
    for a, b in combinations(algorithms, 2):
        try:
            w, p_raw = wilcoxon(wide_df[a], wide_df[b])
        except ValueError:
            w, p_raw = np.nan, np.nan
        rows.append({"comparison": f"{label(a)} vs {label(b)}", "W": w, "p_raw": p_raw})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_bonferroni(out["p_raw"].to_numpy())
    out["significant"] = out["p_holm"] < alpha
    return out


def rank_and_friedman(wide_df, algorithms, alpha=0.05, ascending=True, names=None):
    """Rank / Friedman / all-pairs Wilcoxon+Holm, for any (dataset x algorithm)
    wide metric table -- test MSE, tuning wall-clock, or anything else.

    This is the procedure of Demsar (2006) that the paper's Statistical Procedure
    section describes. Per-dataset error scales are incomparable, so the methods
    are ranked within each dataset, the Friedman omnibus test is applied across
    the dataset-blocks, and where it rejects, all-pairs Wilcoxon signed-rank
    tests follow with a Holm step-down correction.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Index = dataset, columns include every name in ``algorithms``; values are
        the per-(dataset, algorithm) aggregate the caller already computed (the
        paper uses the mean across the ten seeds).
    algorithms : list of str
    alpha : float
    ascending : bool
        Whether a lower value is better (MSE, wall-clock) or, when False, a
        higher value is better. Only the rank direction changes -- rank 1 always
        means "best" in the returned ``rank_matrix``/``avg_rank``.
    names : dict, optional
        Display names for the post-hoc ``comparison`` column.

    Returns
    -------
    dict with keys: rank_matrix, friedman_stat, p_friedman, significant,
    avg_rank, posthoc (a DataFrame, or None if the omnibus test did not reject).
    """
    # method="average" splits ties evenly, matching how scipy.stats.friedmanchisquare
    # ranks internally, so the manual rank matrix and the omnibus test agree.
    rank_matrix = wide_df[algorithms].rank(axis=1, method="average", ascending=ascending)

    friedman_stat, p_friedman = friedmanchisquare(*[wide_df[a] for a in algorithms])
    significant = p_friedman < alpha

    avg_rank = rank_matrix.mean(axis=0).sort_values()

    posthoc = all_pairs_wilcoxon(wide_df, algorithms, alpha=alpha, names=names) if significant else None

    return dict(
        rank_matrix=rank_matrix, friedman_stat=friedman_stat, p_friedman=p_friedman,
        significant=significant, avg_rank=avg_rank, posthoc=posthoc,
    )
