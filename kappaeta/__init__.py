"""
kappaeta — Distance-weighted encoding and regression with gradient-based optimisation.

Provides scikit-learn compatible estimators for kappa-eta encoding/prediction,
JAX-differentiable loss and gradient computation, and Adam / L-BFGS optimisers.
"""

from .estimators import EtaRegressor, KappaEncoder, KappaEtaRegressor
from .jax_ops import (
    compute_gradients_jax,
    grad_loss_jax,
    loss_function_jax,
    pipeline_forward_jax,
)
from .optimizers import AdamOptimizer, LBFGSOptimizer

__all__ = [
    # Estimators
    "KappaEncoder",
    "EtaRegressor",
    "KappaEtaRegressor",
    # JAX operations
    "pipeline_forward_jax",
    "loss_function_jax",
    "grad_loss_jax",
    "compute_gradients_jax",
    # Optimizers
    "AdamOptimizer",
    "LBFGSOptimizer",
]
