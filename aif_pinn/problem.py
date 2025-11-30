"""Problem configuration objects for AIF-PINN experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from torch import Tensor


@dataclass
class ProblemConfig:
    """Configuration container for the SPFDE derived from the Solow-Swan model."""

    alpha: float = 0.8
    epsilon: float = 0.05
    lambda_coeff: float = 1.0
    initial_condition: float = -0.5
    horizon: float = 1.0

    @property
    def boundary_layer_extent(self) -> float:
        """Width of the inner layer that requires refined sampling."""

        return 5.0 * self.epsilon

    def rhs(self, x: Tensor) -> Tensor:
        """Right-hand side forcing term f(x); override for custom scenarios."""

        if not torch.is_tensor(x):
            raise TypeError("rhs expects a torch.Tensor input.")
        return torch.zeros_like(x)


__all__ = ["ProblemConfig"]
