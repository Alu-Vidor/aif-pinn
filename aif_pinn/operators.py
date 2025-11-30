"""Fractional derivative operators and related utilities."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class FractionalDerivativeOperator:
    """Caputo fractional derivative discretization via the L1 scheme."""

    def __init__(
        self,
        grid: np.ndarray,
        alpha: float,
        *,
        dtype: Optional["torch.dtype"] = None,
        device: Optional["torch.device"] = None,
    ) -> None:
        self.alpha = float(alpha)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1) for the Caputo derivative.")

        flat_grid = np.asarray(grid, dtype=np.float64).reshape(-1)
        if flat_grid.size < 2:
            raise ValueError("grid must contain at least two collocation points.")

        if np.any(np.diff(flat_grid) <= 0.0):
            raise ValueError("grid must be strictly increasing for the L1 scheme.")

        self.grid = flat_grid
        self._precomputed_matrix = self.compute_matrix_P()
        self._matrix_cache: dict[tuple[Optional["torch.dtype"], Optional["torch.device"]], "torch.Tensor"] = {}
        self._preferred_dtype = dtype
        self._preferred_device = device

    def compute_matrix_P(self) -> np.ndarray:
        """Construct the dense L1 differentiation matrix once before training."""

        n_points = self.grid.size
        matrix = np.zeros((n_points, n_points), dtype=np.float64)
        inv_gamma = 1.0 / math.gamma(2.0 - self.alpha)

        for j in range(1, n_points):
            x_j = self.grid[j]
            for k in range(1, j + 1):
                delta = self.grid[k] - self.grid[k - 1]
                weight = (pow(x_j - self.grid[k - 1], 1.0 - self.alpha) - pow(x_j - self.grid[k], 1.0 - self.alpha)) / delta
                coeff = inv_gamma * weight
                matrix[j, k] += coeff
                matrix[j, k - 1] -= coeff

        return matrix

    def _get_torch_matrix(self, *, dtype, device):
        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - informative error
            raise ImportError("PyTorch is required to apply the fractional operator.") from exc

        target_dtype = dtype or self._preferred_dtype or torch.float32
        target_device = device or self._preferred_device

        cache_key = (target_dtype, target_device)
        if cache_key not in self._matrix_cache:
            self._matrix_cache[cache_key] = torch.as_tensor(
                self._precomputed_matrix,
                dtype=target_dtype,
                device=target_device,
            )

        return self._matrix_cache[cache_key]

    def apply(self, u_tensor):
        """Apply the precomputed fractional differentiation matrix to u."""

        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - informative error
            raise ImportError("PyTorch is required to apply the fractional operator.") from exc

        if u_tensor.shape[0] != self._precomputed_matrix.shape[0]:
            raise ValueError("u_tensor must have the same leading dimension as the collocation grid.")

        matrix = self._get_torch_matrix(dtype=u_tensor.dtype, device=u_tensor.device)
        return torch.matmul(matrix, u_tensor)


__all__ = ["FractionalDerivativeOperator"]
