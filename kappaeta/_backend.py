"""Backend resolution for kappaeta's dual MLX/CPU implementations.

Two backends are available:

``"mlx"``
    Runs on the Apple GPU via MLX, and is therefore Apple Silicon only. This is
    the backend the optimizer experiments in the paper are measured on.
``"cpu"``
    The portable fallback, available everywhere. Which library serves it depends
    on the operation: the encoder and regressor kernels
    (:mod:`kappaeta._numba`) are numba-compiled, while the differentiable loss
    and gradient used by the optimizers come from :mod:`kappaeta.jax_ops`.

Both backends compute the same quantities in log space and are checked against
each other by ``tests/test_backend_parity.py``.
"""

try:
    import mlx.core as mx  # noqa: F401

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

DEFAULT_BACKEND = "mlx" if MLX_AVAILABLE else "cpu"

_VALID_BACKENDS = ("mlx", "cpu")


def resolve_backend(backend):
    """Resolve a user-supplied ``backend`` argument to a concrete backend name.

    Parameters
    ----------
    backend : {"mlx", "cpu", None}
        ``None`` resolves to :data:`DEFAULT_BACKEND`.

    Returns
    -------
    str
        Either ``"mlx"`` or ``"cpu"``.

    Raises
    ------
    ValueError
        If ``backend`` is not ``None``, ``"mlx"``, or ``"cpu"``.
    ImportError
        If ``backend="mlx"`` is requested but mlx is not installed.
    """
    resolved = DEFAULT_BACKEND if backend is None else backend
    if resolved not in _VALID_BACKENDS:
        raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {resolved!r}")
    if resolved == "mlx" and not MLX_AVAILABLE:
        raise ImportError(
            "backend='mlx' requested but mlx is not installed "
            "(mlx is only available on Apple Silicon macOS)."
        )
    return resolved
