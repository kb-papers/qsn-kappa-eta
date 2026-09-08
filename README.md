# Quasi-Newton Methods for Kappa-Eta Regression

Code supporting the paper's empirical results: gradient validation, an optimizer
comparison (Adam, L-BFGS, SciPy's L-BFGS-B, and a grid-search baseline) across a
20-dataset panel, and the epsilon-sensitivity and scaling analyses of the appendix.

## Layout

- `kappaeta/` -- the estimators (`KappaEncoder`, `EtaRegressor`, `KappaEtaRegressor`),
  the JAX and MLX differentiable loss/gradient implementations, and the Adam and
  L-BFGS optimizers. `backend="mlx"|"cpu"` selects the compute path; MLX is used
  where available (Apple Silicon) and CPU (Numba) otherwise.
- `datasets.py` -- the 20-dataset panel (5 synthetic, 15 from the UCI repository)
  and the tuning grids/initializations every optimizer is given.
- `analytical_derivatives.py` -- closed-form gradients used to validate the
  automatic-differentiation path.
- `methodology.py` -- the shared experiment protocol: train/val/test splitting,
  the t_max-budgeted search, the grid-search baseline, and the
  rank/Friedman/Wilcoxon-Holm significance procedure.
- `01_gradient_validation.ipynb` -- validates AD gradients against the analytical
  and finite-difference references.
- `02_optimizer_comparison_run.ipynb` -- runs the search across the dataset panel
  and writes per-run results to `results/`.
- `03_results_and_significance.ipynb` -- reads those results and produces the
  paper's Results and Discussion analysis.
- `scripts/sensitivity_and_scaling.py` -- the epsilon-sensitivity sweep and the
  gradient/Hessian scaling study of the appendix; writes its measurements to
  `results/`.
- `results/` -- CSV outputs from the notebooks and script above.
- `tests/` -- backend parity and unit tests (`pytest -m network` additionally
  runs parity tests against real UCI datasets; deselected by default).

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

MLX is installed automatically on Apple Silicon (`sys_platform == 'darwin' and
platform_machine == 'arm64'`); everywhere else the estimators fall back to the
CPU (Numba) backend.

## Reproducing the results

Run the notebooks in order (01, 02, 03), or run `scripts/sensitivity_and_scaling.py`
for the appendix's sensitivity and scaling analyses:

```
uv run jupyter nbconvert --to notebook --execute 01_gradient_validation.ipynb
uv run jupyter nbconvert --to notebook --execute 02_optimizer_comparison_run.ipynb
uv run jupyter nbconvert --to notebook --execute 03_results_and_significance.ipynb
uv run python scripts/sensitivity_and_scaling.py
```

## Tests

```
uv run pytest
```
