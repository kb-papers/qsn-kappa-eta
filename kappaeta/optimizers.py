"""Gradient-based optimizers for kappa-eta hyperparameter selection.

``AdamOptimizer`` is the paper's first-order method; ``LBFGSOptimizer`` is its
quasi-Newton method, enforcing the parameter bounds by clipping after each step. The third method compared in the paper, L-BFGS-B, is
SciPy's and is wrapped in ``methodology.run_lbfgsb``.

Both consume automatic-differentiation gradients: MLX's under ``backend="mlx"``,
JAX's under ``backend="cpu"``.
"""

import logging
import time

import jax
import jax.numpy as jnp

from ._backend import resolve_backend
from .jax_ops import compute_loss_and_grad_jax, loss_function_jax

try:
    import mlx.core as mx

    from . import mlx_ops
except ImportError:
    mx = None
    mlx_ops = None

logger = logging.getLogger(__name__)


def _dot(a, b):
    """Backend-agnostic dot product (works for both jnp and mx arrays)."""
    return (a * b).sum()


class _BaseOptimizer:
    """Shared utilities for kappa-eta optimizers.

    Parameters
    ----------
    backend : {"mlx", "cpu", None}
        Which automatic-differentiation engine computes the loss and gradient
        every iteration. ``None`` (default) resolves to MLX if installed
        (Apple Silicon), else to JAX. Note that ``"cpu"`` selects JAX here,
        whereas for the estimators it selects the numba kernels.
    time_budget : float, optional
        Maximum wall-clock time in seconds for the optimisation loop.
        ``None`` (default) means no time limit.
    """

    def __init__(
        self, *, max_iters, tol, kappa_bounds, eta_bounds, verbose, backend=None,
        time_budget=None,
    ):
        self.max_iters = max_iters
        self.tol = tol
        self.kappa_bounds = kappa_bounds
        self.eta_bounds = eta_bounds
        self.verbose = verbose
        self.backend = resolve_backend(backend)
        self.time_budget = time_budget
        self.history: list[dict] = []

        if self.backend == "mlx":
            self._xp = mx
            self._loss_fn = mlx_ops.loss_function_mlx
            self._loss_and_grad_fn = mlx_ops.compute_loss_and_grad_mlx
            self._grad_transform = mx.grad
            self._value_and_grad_transform = mx.value_and_grad
        else:
            # backend == "cpu": there is no numba kernel for the gradient
            # step, so the CPU path uses JAX's reverse-mode differentiation.
            self._xp = jnp
            self._loss_fn = loss_function_jax
            self._loss_and_grad_fn = compute_loss_and_grad_jax
            self._grad_transform = jax.grad
            self._value_and_grad_transform = jax.value_and_grad

    def clip_params(self, kappa, eta):
        """Clip parameters to valid bounds."""
        xp = self._xp
        kappa_clipped = xp.clip(kappa, self.kappa_bounds[0], self.kappa_bounds[1])
        eta_clipped = xp.clip(eta, self.eta_bounds[0], self.eta_bounds[1])
        return kappa_clipped, eta_clipped

    def _to_array(self, *arrays):
        return tuple(self._xp.array(a) for a in arrays)

    def _loss_converged(self, prev_loss, loss):
        """Relative loss-change convergence test, matching scipy L-BFGS-B's ftol:
        ``|f_k - f_{k+1}| <= tol * max(|f_k|, |f_{k+1}|, 1)``.

        The test is relative rather than absolute because an absolute
        ``|prev_loss - loss| < tol`` is unreachable once the loss scale is much
        bigger than tol -- a dataset with MSE in the thousands would never
        satisfy ``< 1e-8`` and would always burn the full iteration budget.
        ``abs()`` on the left-hand side (scipy uses the signed
        ``f_k - f_{k+1}``) avoids declaring convergence on an iteration where
        the loss ticks up, since Adam has no line search to guarantee monotonic
        decrease.
        """
        scale = max(abs(prev_loss), abs(loss), 1.0)
        return abs(prev_loss - loss) < self.tol * scale


class AdamOptimizer(_BaseOptimizer):
    """Adam optimizer with parameter clipping.

    Parameters
    ----------
    learning_rate : float
        Step size.
    beta1, beta2 : float
        Exponential decay rates for the first and second moment estimates.
    epsilon : float
        Small constant for numerical stability.
    max_iters : int
        Maximum number of iterations.
    tol : float
        Relative convergence tolerance on loss change:
        ``|prev_loss - loss| < tol * max(|prev_loss|, |loss|, 1)`` (matches
        scipy L-BFGS-B's ``ftol``).
    kappa_bounds, eta_bounds : tuple of (float, float)
        Box constraints for the parameters.
    verbose : bool
        Whether to log progress.
    backend : {"mlx", "cpu", None}
        Which implementation computes the loss/gradient every iteration.
    time_budget : float, optional
        Maximum wall-clock time in seconds for the optimisation loop.
        ``None`` (default) means no time limit.
    """

    def __init__(
        self,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        max_iters=100,
        tol=1e-6,
        kappa_bounds=(0.1, 10.0),
        eta_bounds=(0.1, 10.0),
        verbose=True,
        backend=None,
        time_budget=None,
    ):
        super().__init__(
            max_iters=max_iters,
            tol=tol,
            kappa_bounds=kappa_bounds,
            eta_bounds=eta_bounds,
            verbose=verbose,
            backend=backend,
            time_budget=time_budget,
        )
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def optimize(self, kappa_init, eta_init, X_eval, X_train, y_train, y_eval):
        """Run Adam optimisation.

        Returns
        -------
        kappa_opt, eta_opt, history
        """
        xp = self._xp
        X_eval_a, X_train_a, y_train_a, y_eval_a = self._to_array(
            X_eval, X_train, y_train, y_eval
        )
        kappa = xp.array(kappa_init)
        eta = xp.array(eta_init)

        # Initialise Adam moments
        m_kappa = xp.zeros_like(kappa)
        v_kappa = xp.zeros_like(kappa)
        m_eta = 0.0
        v_eta = 0.0

        self.history = []
        prev_loss = float("inf")

        # Eval-effort counters. Adam has no line search, so n_linesearch is
        # always 0; nfev == njev == nit since every iteration does exactly
        # one loss and one gradient evaluation.
        self.nfev = 0
        self.njev = 0
        self.n_linesearch = 0
        self.nit = 0
        self.stop_reason = "max_iters"
        start_time = time.perf_counter()

        for iteration in range(1, self.max_iters + 1):
            loss, grad_kappa, grad_eta = self._loss_and_grad_fn(
                kappa, eta, X_eval_a, X_train_a, y_train_a, y_eval_a
            )
            self.nfev += 1
            self.njev += 1

            # Update biased moment estimates
            m_kappa = self.beta1 * m_kappa + (1 - self.beta1) * grad_kappa
            m_eta = self.beta1 * m_eta + (1 - self.beta1) * grad_eta
            v_kappa = self.beta2 * v_kappa + (1 - self.beta2) * (grad_kappa ** 2)
            v_eta = self.beta2 * v_eta + (1 - self.beta2) * (grad_eta ** 2)

            # Bias-corrected estimates
            m_kappa_hat = m_kappa / (1 - self.beta1 ** iteration)
            m_eta_hat = m_eta / (1 - self.beta1 ** iteration)
            v_kappa_hat = v_kappa / (1 - self.beta2 ** iteration)
            v_eta_hat = v_eta / (1 - self.beta2 ** iteration)

            self.history.append(
                {
                    "iteration": iteration - 1,
                    "loss": float(loss),
                    "kappa": float(kappa) if kappa.ndim == 0 else kappa.tolist(),
                    "eta": float(eta),
                    "grad_kappa": (
                        float(grad_kappa)
                        if grad_kappa.ndim == 0
                        else grad_kappa.tolist()
                    ),
                    "grad_eta": float(grad_eta),
                }
            )

            if self.verbose and (iteration - 1) % 10 == 0:
                logger.info(
                    "Iter %d: Loss=%.6f, kappa=%s, eta=%.4f",
                    iteration - 1, loss, kappa, float(eta),
                )

            # Convergence check
            if self._loss_converged(prev_loss, float(loss)):
                self.stop_reason = "converged"
                if self.verbose:
                    logger.info("Converged at iteration %d", iteration - 1)
                break

            # Time budget check
            if self.time_budget is not None and (
                time.perf_counter() - start_time
            ) > self.time_budget:
                self.stop_reason = "time_budget"
                if self.verbose:
                    logger.info(
                        "Time budget exceeded at iteration %d (%.2fs elapsed)",
                        iteration - 1, time.perf_counter() - start_time,
                    )
                break

            # Guard against NaN gradients
            if xp.any(xp.isnan(grad_kappa)) or xp.isnan(grad_eta):
                if self.verbose:
                    logger.warning("NaN gradient at iter %d, skipping update", iteration - 1)
                if xp.any(xp.isnan(grad_kappa)):
                    m_kappa = xp.zeros_like(kappa)
                    v_kappa = xp.zeros_like(kappa)
                if xp.isnan(grad_eta):
                    m_eta = 0.0
                    v_eta = 0.0
                continue

            prev_loss = float(loss)

            # Adam parameter update
            kappa = kappa - self.learning_rate * m_kappa_hat / (
                xp.sqrt(v_kappa_hat) + self.epsilon
            )
            eta = eta - self.learning_rate * m_eta_hat / (
                xp.sqrt(v_eta_hat) + self.epsilon
            )
            kappa, eta = self.clip_params(kappa, eta)

        self.nit = len(self.history)

        if self.verbose:
            logger.info("Final: Loss=%.6f, kappa=%s, eta=%.4f", loss, kappa, float(eta))

        return kappa, eta, self.history


class LBFGSOptimizer(_BaseOptimizer):
    """L-BFGS (Limited-memory BFGS) optimizer.

    Approximates the inverse Hessian using a history of gradient differences,
    giving quasi-Newton convergence at O(n) per-iteration cost instead of O(n³).

    Parameters
    ----------
    max_iters : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on gradient norm, and relative tolerance on
        loss change: ``|prev_loss - loss| < tol * max(|prev_loss|, |loss|, 1)``
        (matches scipy L-BFGS-B's ``ftol``).
    memory_size : int
        Number of past (s, y) pairs to store.
    kappa_bounds, eta_bounds : tuple of (float, float)
        Box constraints for the parameters.
    initial_step_size : float
        Initial line-search step size.
    verbose : bool
        Whether to log progress.
    backend : {"mlx", "cpu", None}
        Which implementation computes the loss/gradient every iteration.
    time_budget : float, optional
        Maximum wall-clock time in seconds for the optimisation loop.
        ``None`` (default) means no time limit.
    """

    def __init__(
        self,
        max_iters=200,
        tol=1e-6,
        memory_size=10,
        kappa_bounds=(0.1, 10.0),
        eta_bounds=(0.1, 10.0),
        initial_step_size=1.0,
        verbose=True,
        backend=None,
        time_budget=None,
    ):
        super().__init__(
            max_iters=max_iters,
            tol=tol,
            kappa_bounds=kappa_bounds,
            eta_bounds=eta_bounds,
            verbose=verbose,
            backend=backend,
            time_budget=time_budget,
        )
        self.memory_size = memory_size
        self.initial_step_size = initial_step_size

    def _pack(self, kappa, eta):
        xp = self._xp
        if kappa.ndim == 0:
            return xp.stack([kappa, eta])
        return xp.concatenate([kappa, xp.reshape(eta, (1,))])

    @staticmethod
    def _unpack(param_vec, n_kappa, scalar_kappa=False):
        kappa = param_vec[:n_kappa]
        eta = param_vec[n_kappa]
        if scalar_kappa and n_kappa == 1:
            kappa = kappa[0]
        return kappa, eta

    def _lbfgs_direction(self, grad_flat, s_history, y_history):
        """Compute L-BFGS search direction via two-loop recursion."""
        xp = self._xp
        q = grad_flat
        m = len(s_history)

        if m == 0:
            return -grad_flat

        alphas = []
        rhos = []

        # Backward loop
        for i in range(m - 1, -1, -1):
            s_i = s_history[i]
            y_i = y_history[i]
            rho_i = 1.0 / (_dot(y_i, s_i) + 1e-12)
            rhos.insert(0, rho_i)
            alpha_i = rho_i * _dot(s_i, q)
            alphas.insert(0, alpha_i)
            q = q - alpha_i * y_i

        # Initial Hessian scaling: H_0 = gamma * I
        s_last = s_history[-1]
        y_last = y_history[-1]
        gamma = _dot(s_last, y_last) / (_dot(y_last, y_last) + 1e-12)
        gamma = xp.maximum(gamma, 1e-6)
        r = gamma * q

        # Forward loop
        for i in range(m):
            beta = rhos[i] * _dot(y_history[i], r)
            r = r + s_history[i] * (alphas[i] - beta)

        return -r


    def optimize(self, kappa_init, eta_init, X_eval, X_train, y_train, y_eval):
        """Run L-BFGS optimisation.

        Returns
        -------
        kappa_opt, eta_opt, history
        """
        xp = self._xp
        X_eval_a, X_train_a, y_train_a, y_eval_a = self._to_array(
            X_eval, X_train, y_train, y_eval
        )
        kappa = xp.array(kappa_init)
        eta = xp.array(eta_init)
        n_kappa = kappa.shape[0] if kappa.ndim > 0 else 1
        scalar_kappa = kappa.ndim == 0

        def loss_flat(param_vec):
            k = param_vec[:n_kappa]
            e = param_vec[n_kappa]
            if n_kappa == 1:
                k = k[0]
            params = {"kappa": k, "eta": e}
            return self._loss_fn(params, X_eval_a, X_train_a, y_train_a, y_eval_a)

        grad_flat_fn_raw = self._grad_transform(loss_flat)
        loss_and_grad_flat_fn_raw = self._value_and_grad_transform(loss_flat)
        if self.backend == "mlx":
            grad_flat_fn_compiled = mx.compile(grad_flat_fn_raw)

            def grad_flat_fn(param_vec):
                try:
                    g = grad_flat_fn_compiled(param_vec)
                    mx.eval(g)
                    return g
                except ValueError:
                    return grad_flat_fn_raw(param_vec)

            loss_and_grad_flat_fn_compiled = mx.compile(loss_and_grad_flat_fn_raw)

            def loss_and_grad_flat_fn(param_vec):
                try:
                    loss_val, g = loss_and_grad_flat_fn_compiled(param_vec)
                    mx.eval(loss_val, g)
                    return loss_val, g
                except ValueError:
                    return loss_and_grad_flat_fn_raw(param_vec)
        else:
            grad_flat_fn = grad_flat_fn_raw
            loss_and_grad_flat_fn = loss_and_grad_flat_fn_raw

        self.nfev = 0
        self.njev = 0
        self.n_linesearch = 0
        self.nit = 0
        self.stop_reason = "max_iters"
        start_time = time.perf_counter()

        x = self._pack(kappa, eta)
        loss, g = loss_and_grad_flat_fn(x)
        self.nfev += 1
        self.njev += 1

        s_history = []
        y_history = []

        self.history = []
        prev_loss = float("inf")

        for iteration in range(self.max_iters):
            grad_norm = float(xp.linalg.norm(g))
            kappa_cur, eta_cur = self._unpack(x, n_kappa, scalar_kappa)

            self.history.append(
                {
                    "iteration": iteration,
                    "loss": float(loss),
                    "kappa": (
                        float(kappa_cur) if kappa_cur.ndim == 0 else kappa_cur.tolist()
                    ),
                    "eta": float(eta_cur),
                    "grad_norm": grad_norm,
                }
            )

            if self.verbose and iteration % 10 == 0:
                logger.info(
                    "L-BFGS Iter %d: Loss=%.6f, |grad|=%.6f", iteration, loss, grad_norm
                )

            # Convergence check
            if self._loss_converged(prev_loss, float(loss)) or grad_norm < self.tol:
                self.stop_reason = "converged"
                if self.verbose:
                    logger.info("L-BFGS converged at iteration %d", iteration)
                break

            # Time budget check
            if self.time_budget is not None and (
                time.perf_counter() - start_time
            ) > self.time_budget:
                self.stop_reason = "time_budget"
                if self.verbose:
                    logger.info(
                        "L-BFGS time budget exceeded at iteration %d (%.2fs elapsed)",
                        iteration, time.perf_counter() - start_time,
                    )
                break

            if xp.any(xp.isnan(g)):
                self.stop_reason = "nan_gradient"
                if self.verbose:
                    logger.warning("NaN gradient at iter %d", iteration)
                break

            # Search direction
            direction = self._lbfgs_direction(g, s_history, y_history)

            # Ensure descent direction
            directional_deriv = _dot(g, direction)
            if directional_deriv > 0:
                direction = -g
                directional_deriv = _dot(g, direction)

            # Backtracking line search (Armijo condition)
            alpha = self.initial_step_size
            c1 = 1e-4

            x_new = x + alpha * direction
            kappa_new, eta_new = self._unpack(x_new, n_kappa, scalar_kappa)
            kappa_new, eta_new = self.clip_params(kappa_new, eta_new)
            x_new = self._pack(kappa_new, eta_new)
            loss_new = loss_flat(x_new)
            self.nfev += 1

            backtracks = 0
            while (
                (xp.isnan(loss_new) or loss_new > loss + c1 * alpha * directional_deriv)
                and alpha > 1e-6
                and backtracks < 20
            ):
                alpha *= 0.5
                x_new = x + alpha * direction
                kappa_new, eta_new = self._unpack(x_new, n_kappa, scalar_kappa)
                kappa_new, eta_new = self.clip_params(kappa_new, eta_new)
                x_new = self._pack(kappa_new, eta_new)
                loss_new = loss_flat(x_new)
                self.nfev += 1
                backtracks += 1
            self.n_linesearch += backtracks

            g_new = grad_flat_fn(x_new)
            self.njev += 1
            s_k = x_new - x
            y_k = g_new - g

            curvature = _dot(s_k, y_k)
            if curvature > 1e-10:
                s_history.append(s_k)
                y_history.append(y_k)
                if len(s_history) > self.memory_size:
                    s_history.pop(0)
                    y_history.pop(0)

            prev_loss = float(loss)
            x = x_new
            loss = loss_new
            g = g_new

        # Final parameters
        self.nit = len(self.history)
        kappa_final, eta_final = self._unpack(x, n_kappa, scalar_kappa)
        params = {"kappa": kappa_final, "eta": eta_final}
        loss = self._loss_fn(params, X_eval_a, X_train_a, y_train_a, y_eval_a)
        self.nfev += 1

        if self.history:
            self.history[-1]["loss"] = float(loss)

        if self.verbose:
            logger.info(
                "L-BFGS Final: Loss=%.6f, kappa=%s, eta=%.4f",
                loss, kappa_final, float(eta_final),
            )

        return kappa_final, eta_final, self.history
