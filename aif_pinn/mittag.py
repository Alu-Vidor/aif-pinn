"""Mittag-Leffler function approximations used by the PINN models."""

from __future__ import annotations

import math

import torch


def mittag_leffler_approx(
    z: torch.Tensor,
    alpha: float,
    *,
    series_terms: int = 6,
    switch_threshold: float = 1.0,
) -> torch.Tensor:
    """Piecewise approximation of Mittag-Leffler."""

    if series_terms <= 0:
        raise ValueError("series_terms must be positive.")
    if not torch.is_tensor(z):
        raise TypeError("z must be a torch.Tensor.")

    abs_z = torch.abs(z)
    result = torch.zeros_like(z)
    small_mask = abs_z <= switch_threshold

    if small_mask.any():
        small_z = z[small_mask]
        series = torch.zeros_like(small_z)
        denominators = [math.gamma(alpha * k + 1.0) for k in range(series_terms)]
        for k, denom in enumerate(denominators):
            series = series + small_z.pow(k) / denom
        result[small_mask] = series

    if (~small_mask).any():
        large_z = z[~small_mask]
        safe_large = torch.where(
            torch.abs(large_z) < 1e-8,
            torch.full_like(large_z, 1e-8),
            large_z,
        )
        gamma_factor = math.gamma(1.0 - alpha)
        result[~small_mask] = -1.0 / (safe_large * gamma_factor)

    return result


__all__ = ["mittag_leffler_approx"]
