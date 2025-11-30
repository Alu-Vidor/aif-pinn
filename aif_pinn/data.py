"""Data generation utilities for collocation grids."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .problem import ProblemConfig


class DataGenerator:
    """Generate collocation points tailored for boundary layer behaviour."""

    _MIN_POSITIVE = np.finfo(float).tiny

    def __init__(self, config: ProblemConfig) -> None:
        self.config = config

    def generate_grid(self, num_points: int) -> np.ndarray:
        """Create a log-uniform + uniform grid for the SPFDE collocation points."""

        if num_points < 2:
            raise ValueError("num_points must be at least 2 to split between regions.")

        layer_extent = min(self.config.boundary_layer_extent, self.config.horizon)
        if layer_extent <= 0.0:
            return np.linspace(0.0, self.config.horizon, num=num_points)

        n_layer = num_points // 2
        n_outer = num_points - n_layer

        layer_points = self._log_uniform_layer(layer_extent, n_layer)
        outer_points = self._uniform_outer(layer_extent, n_outer)

        return np.concatenate([layer_points, outer_points])

    def to_torch(
        self,
        grid: np.ndarray,
        *,
        dtype: Optional["torch.dtype"] = None,
        device: Optional["torch.device"] = None,
    ):
        """Convert a numpy grid to a torch tensor without enforcing a dependency."""

        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - informative error
            raise ImportError("PyTorch is required to call to_torch.") from exc

        target_dtype = dtype or torch.float32
        return torch.as_tensor(grid, dtype=target_dtype, device=device).reshape(-1, 1)

    def to_jax(self, grid: np.ndarray, *, dtype: Optional["jnp.dtype"] = None):
        """Convert a numpy grid to a JAX array."""

        try:
            import jax.numpy as jnp  # type: ignore
        except ImportError as exc:  # pragma: no cover - informative error
            raise ImportError("JAX is required to call to_jax.") from exc

        target_dtype = dtype or jnp.float32
        return jnp.asarray(grid, dtype=target_dtype).reshape(-1, 1)

    def _log_uniform_layer(self, layer_extent: float, n_layer: int) -> np.ndarray:
        """Generate log-uniform points inside the inner boundary layer."""

        if n_layer <= 0:
            return np.empty(0, dtype=np.float64)
        if n_layer == 1:
            return np.array([0.0], dtype=np.float64)

        positive_min = max(self._MIN_POSITIVE, layer_extent * 1e-6)
        log_points = np.logspace(
            np.log10(positive_min),
            np.log10(layer_extent),
            num=n_layer,
            endpoint=False,
            dtype=np.float64,
        )
        trimmed = log_points[: n_layer - 1]

        return np.concatenate(([0.0], trimmed))

    def _uniform_outer(self, layer_extent: float, n_outer: int) -> np.ndarray:
        """Generate uniform points in the outer domain, starting at the layer edge."""

        if n_outer <= 0:
            return np.empty(0, dtype=np.float64)

        if layer_extent >= self.config.horizon:
            return np.linspace(0.0, self.config.horizon, num=n_outer, endpoint=True, dtype=np.float64)

        return np.linspace(layer_extent, self.config.horizon, num=n_outer, endpoint=True, dtype=np.float64)


__all__ = ["DataGenerator"]
