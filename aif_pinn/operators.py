"""Fractional derivative operators and related utilities."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


class FractionalDerivativeOperator:
    """Caputo fractional derivative discretization via the L1 scheme."""

    def __init__(
        self,
        grid: np.ndarray,
        alpha: float,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
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
        self._matrix_np: Optional[np.ndarray] = None
        self.matrix_tensor: Optional[torch.Tensor] = None

        matrix = self.compute_matrix_P()
        target_dtype = dtype or torch.float32
        target_device = device or torch.device("cpu")
        self.matrix_tensor = torch.as_tensor(matrix, dtype=target_dtype, device=target_device)

    def compute_matrix_P(self) -> np.ndarray:
        """Construct the dense L1 differentiation matrix once before training."""

        if self._matrix_np is not None:
            return self._matrix_np

        x = self.grid
        n_points = x.size
        inv_gamma = 1.0 / math.gamma(2.0 - self.alpha)
        n_intervals = n_points - 1

        if n_intervals <= 0:
            raise ValueError("grid must contain at least two points.")

        rows = x[1:].reshape(-1, 1)
        left_nodes = x[:-1].reshape(1, -1)
        right_nodes = x[1:].reshape(1, -1)
        mask = np.tri(n_intervals, dtype=bool)

        delta = np.diff(x).reshape(1, -1)

        diff_left = np.where(mask, rows - left_nodes, 0.0)
        diff_right = np.where(mask, rows - right_nodes, 0.0)
        power = 1.0 - self.alpha
        numerator = np.power(diff_left, power) - np.power(diff_right, power)
        weight = numerator / delta
        coeff = inv_gamma * weight

        matrix = np.zeros((n_points, n_points), dtype=np.float64)
        matrix[1:, 1:] = coeff
        matrix[1:, :-1] -= coeff

        self._matrix_np = matrix
        return matrix

    def apply(self, u_tensor: torch.Tensor) -> torch.Tensor:
        """Apply the precomputed fractional differentiation matrix to u."""

        if self.matrix_tensor is None:
            raise RuntimeError("The fractional operator matrix tensor is not initialized.")

        if u_tensor.shape[0] != self.matrix_tensor.shape[0]:
            raise ValueError("u_tensor must have the same leading dimension as the collocation grid.")

        return torch.matmul(self.matrix_tensor, u_tensor)


__all__ = ["FractionalDerivativeOperator"]
