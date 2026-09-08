"""Unit tests for kappaeta._backend. Runs on all platforms (no mlx required)."""

import pytest

from kappaeta import _backend


def test_default_backend_matches_availability():
    expected = "mlx" if _backend.MLX_AVAILABLE else "cpu"
    assert _backend.DEFAULT_BACKEND == expected


def test_resolve_backend_none_uses_default():
    assert _backend.resolve_backend(None) == _backend.DEFAULT_BACKEND


def test_resolve_backend_cpu_always_valid():
    assert _backend.resolve_backend("cpu") == "cpu"


def test_resolve_backend_invalid_raises():
    with pytest.raises(ValueError):
        _backend.resolve_backend("bogus")


def test_resolve_backend_mlx_unavailable_raises(monkeypatch):
    monkeypatch.setattr(_backend, "MLX_AVAILABLE", False)
    with pytest.raises(ImportError):
        _backend.resolve_backend("mlx")


def test_resolve_backend_mlx_available_when_installed():
    if not _backend.MLX_AVAILABLE:
        pytest.skip("mlx not installed on this machine")
    assert _backend.resolve_backend("mlx") == "mlx"
