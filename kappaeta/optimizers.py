import logging

import jax
import jax.numpy as jnp

from .jax_ops import compute_gradients_jax, loss_function_jax

logger = logging.getLogger(__name__)


class _BaseOptimizer:
    """Shared utilities for kappa-eta optimizers."""

    def __init__(self, *, max_iters, tol, kappa_bounds, eta_bounds, verbose):
        self.max_iters = max_iters
        self.tol = tol
        self.kappa_bounds = kappa_bounds
        self.eta_bounds = eta_bounds
        self.verbose = verbose
        self.history: list[dict] = []

    def clip_params(self, kappa, eta):
        """Clip parameters to valid bounds."""
        kappa_clipped = jnp.clip(kappa, self.kappa_bounds[0], self.kappa_bounds[1])
        eta_clipped = jnp.clip(eta, self.eta_bounds[0], self.eta_bounds[1])
        return kappa_clipped, eta_clipped

    @staticmethod
    def _to_jax(*arrays):
        return tuple(jnp.array(a) for a in arrays)


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
        Convergence tolerance on loss change.
    kappa_bounds, eta_bounds : tuple of (float, float)
        Box constraints for the parameters.
    verbose : bool
        Whether to log progress.
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
    ):
        super().__init__(
            max_iters=max_iters,
            tol=tol,
            kappa_bounds=kappa_bounds,
            eta_bounds=eta_bounds,
            verbose=verbose,
        )
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def optimize(self, kappa_init, eta_init, X_test, X_train, y_train, y_test):
        """Run Adam optimisation.

        Returns
        -------
        kappa_opt, eta_opt, history
        """
        X_test_jax, X_train_jax, y_train_jax, y_test_jax = self._to_jax(
            X_test, X_train, y_train, y_test
        )
        kappa = jnp.array(kappa_init)
        eta = jnp.array(eta_init)

        # Initialise Adam moments
        m_kappa = jnp.zeros_like(kappa)
        v_kappa = jnp.zeros_like(kappa)
        m_eta = 0.0
        v_eta = 0.0

        self.history = []
        prev_loss = float("inf")

        for iteration in range(1, self.max_iters + 1):
            params = {"kappa": kappa, "eta": eta}
            loss = loss_function_jax(
                params, X_test_jax, X_train_jax, y_train_jax, y_test_jax
            )
            grad_kappa, grad_eta = compute_gradients_jax(
                kappa, eta, X_test_jax, X_train_jax, y_train_jax, y_test_jax
            )

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
            if abs(prev_loss - float(loss)) < self.tol:
                if self.verbose:
                    logger.info("Converged at iteration %d", iteration - 1)
                break

            # Guard against NaN gradients
            if jnp.any(jnp.isnan(grad_kappa)) or jnp.isnan(grad_eta):
                if self.verbose:
                    logger.warning("NaN gradient at iter %d, skipping update", iteration - 1)
                if jnp.any(jnp.isnan(grad_kappa)):
                    m_kappa = jnp.zeros_like(kappa)
                    v_kappa = jnp.zeros_like(kappa)
                if jnp.isnan(grad_eta):
                    m_eta = 0.0
                    v_eta = 0.0
                continue

            prev_loss = float(loss)

            # Adam parameter update
            kappa = kappa - self.learning_rate * m_kappa_hat / (
                jnp.sqrt(v_kappa_hat) + self.epsilon
            )
            eta = eta - self.learning_rate * m_eta_hat / (
                jnp.sqrt(v_eta_hat) + self.epsilon
            )
            kappa, eta = self.clip_params(kappa, eta)

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
        Convergence tolerance on loss change or gradient norm.
    memory_size : int
        Number of past (s, y) pairs to store.
    kappa_bounds, eta_bounds : tuple of (float, float)
        Box constraints for the parameters.
    initial_step_size : float
        Initial line-search step size.
    verbose : bool
        Whether to log progress.
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
    ):
        super().__init__(
            max_iters=max_iters,
            tol=tol,
            kappa_bounds=kappa_bounds,
            eta_bounds=eta_bounds,
            verbose=verbose,
        )
        self.memory_size = memory_size
        self.initial_step_size = initial_step_size


    @staticmethod
    def _pack(kappa, eta):
        if jnp.ndim(kappa) == 0:
            return jnp.array([kappa, eta])
        return jnp.concatenate([kappa, jnp.array([eta])])

    @staticmethod
    def _unpack(param_vec, n_kappa, scalar_kappa=False):
        kappa = param_vec[:n_kappa]
        eta = param_vec[n_kappa]
        if scalar_kappa and n_kappa == 1:
            kappa = kappa[0]
        return kappa, eta

    @staticmethod
    def _lbfgs_direction(grad_flat, s_history, y_history):
        """Compute L-BFGS search direction via two-loop recursion."""
        q = grad_flat.copy()
        m = len(s_history)

        if m == 0:
            return -grad_flat

        alphas = []
        rhos = []

        # Backward loop
        for i in range(m - 1, -1, -1):
            s_i = s_history[i]
            y_i = y_history[i]
            rho_i = 1.0 / (jnp.dot(y_i, s_i) + 1e-12)
            rhos.insert(0, rho_i)
            alpha_i = rho_i * jnp.dot(s_i, q)
            alphas.insert(0, alpha_i)
            q = q - alpha_i * y_i

        # Initial Hessian scaling: H_0 = gamma * I
        s_last = s_history[-1]
        y_last = y_history[-1]
        gamma = jnp.dot(s_last, y_last) / (jnp.dot(y_last, y_last) + 1e-12)
        gamma = jnp.maximum(gamma, 1e-6)
        r = gamma * q

        # Forward loop
        for i in range(m):
            beta = rhos[i] * jnp.dot(y_history[i], r)
            r = r + s_history[i] * (alphas[i] - beta)

        return -r


    def optimize(self, kappa_init, eta_init, X_test, X_train, y_train, y_test):
        """Run L-BFGS optimisation.

        Returns
        -------
        kappa_opt, eta_opt, history
        """
        X_test_jax, X_train_jax, y_train_jax, y_test_jax = self._to_jax(
            X_test, X_train, y_train, y_test
        )
        kappa = jnp.array(kappa_init)
        eta = jnp.array(eta_init)
        n_kappa = kappa.shape[0] if jnp.ndim(kappa) > 0 else 1
        scalar_kappa = jnp.ndim(kappa) == 0

        def loss_flat(param_vec):
            k = param_vec[:n_kappa]
            e = param_vec[n_kappa]
            if n_kappa == 1:
                k = k[0]
            params = {"kappa": k, "eta": e}
            return loss_function_jax(
                params, X_test_jax, X_train_jax, y_train_jax, y_test_jax
            )

        grad_flat_fn = jax.grad(loss_flat)

        x = self._pack(kappa, eta)
        loss = loss_flat(x)
        g = grad_flat_fn(x)

        s_history: list[jnp.ndarray] = []
        y_history: list[jnp.ndarray] = []

        self.history = []
        prev_loss = float("inf")

        for iteration in range(self.max_iters):
            grad_norm = float(jnp.linalg.norm(g))
            kappa_cur, eta_cur = self._unpack(x, n_kappa, scalar_kappa)

            self.history.append(
                {
                    "iteration": iteration,
                    "loss": float(loss),
                    "kappa": (
                        float(kappa_cur) if jnp.ndim(kappa_cur) == 0 else kappa_cur.tolist()
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
            if abs(prev_loss - float(loss)) < self.tol or grad_norm < self.tol:
                if self.verbose:
                    logger.info("L-BFGS converged at iteration %d", iteration)
                break

            if jnp.any(jnp.isnan(g)):
                if self.verbose:
                    logger.warning("NaN gradient at iter %d", iteration)
                break

            # Search direction
            direction = self._lbfgs_direction(g, s_history, y_history)

            # Ensure descent direction
            directional_deriv = jnp.dot(g, direction)
            if directional_deriv > 0:
                direction = -g
                directional_deriv = jnp.dot(g, direction)

            # Backtracking line search (Armijo condition)
            alpha = self.initial_step_size
            c1 = 1e-4

            x_new = x + alpha * direction
            kappa_new, eta_new = self._unpack(x_new, n_kappa, scalar_kappa)
            kappa_new, eta_new = self.clip_params(kappa_new, eta_new)
            x_new = self._pack(kappa_new, eta_new)
            loss_new = loss_flat(x_new)

            backtracks = 0
            while (
                (jnp.isnan(loss_new) or loss_new > loss + c1 * alpha * directional_deriv)
                and alpha > 1e-6
                and backtracks < 20
            ):
                alpha *= 0.5
                x_new = x + alpha * direction
                kappa_new, eta_new = self._unpack(x_new, n_kappa, scalar_kappa)
                kappa_new, eta_new = self.clip_params(kappa_new, eta_new)
                x_new = self._pack(kappa_new, eta_new)
                loss_new = loss_flat(x_new)
                backtracks += 1

            # Update L-BFGS history
            g_new = grad_flat_fn(x_new)
            s_k = x_new - x
            y_k = g_new - g

            curvature = jnp.dot(s_k, y_k)
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
        kappa_final, eta_final = self._unpack(x, n_kappa, scalar_kappa)
        params = {"kappa": kappa_final, "eta": eta_final}
        loss = loss_function_jax(
            params, X_test_jax, X_train_jax, y_train_jax, y_test_jax
        )

        if self.history:
            self.history[-1]["loss"] = float(loss)

        if self.verbose:
            logger.info(
                "L-BFGS Final: Loss=%.6f, kappa=%s, eta=%.4f",
                loss, kappa_final, float(eta_final),
            )

        return kappa_final, eta_final, self.history
