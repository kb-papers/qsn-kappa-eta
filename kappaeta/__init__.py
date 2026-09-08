"""
kappaeta: distance-weighted encoding and regression with gradient-based optimisation.

Provides scikit-learn compatible estimators for kappa-eta encoding/prediction,
JAX and MLX differentiable loss and gradient computation, and Adam / L-BFGS
optimisers. Estimators and optimisers take a ``backend="mlx"|"cpu"``
argument (default: MLX if installed -- Apple Silicon only -- else "cpu").
"""

from ._backend import DEFAULT_BACKEND, MLX_AVAILABLE
from .estimators import EtaRegressor, KappaEncoder, KappaEtaRegressor
from .jax_ops import (
    compute_gradients_jax,
    compute_loss_and_grad_jax,
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
    "compute_loss_and_grad_jax",
    # Optimizers
    "AdamOptimizer",
    "LBFGSOptimizer",
    # Backend info
    "DEFAULT_BACKEND",
    "MLX_AVAILABLE",
]

# mlx_ops is only importable where mlx is installed (Apple Silicon). Importing
# it defensively here means `import kappaeta` still works everywhere else;
# code that needs it explicitly does `from kappaeta import mlx_ops` (which
# works once mlx_ops has been imported at least once, as it is here) or
# checks kappaeta.MLX_AVAILABLE first.
if MLX_AVAILABLE:
    from . import mlx_ops
    from .mlx_ops import (
        compute_gradients_mlx,
        compute_loss_and_grad_mlx,
        eta_predict_mlx,
        grad_loss_mlx,
        kappa_encode_mlx,
        loss_function_mlx,
        pipeline_forward_mlx,
    )

    __all__ += [
        "mlx_ops",
        "pipeline_forward_mlx",
        "loss_function_mlx",
        "grad_loss_mlx",
        "compute_gradients_mlx",
        "compute_loss_and_grad_mlx",
        "kappa_encode_mlx",
        "eta_predict_mlx",
    ]
