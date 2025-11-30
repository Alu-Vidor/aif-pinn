"""Reference SPFDE problems used for benchmarking."""

from __future__ import annotations

import math
from typing import Optional

import torch

from .mittag import mittag_leffler_approx
from .problem import AbstractSPFDE


class _BaseProblem(AbstractSPFDE):
    """Shared helpers for problems with constant epsilon^alpha factors."""

    def _eps_alpha_tensor(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_tensor(self._eps_alpha)


class LinearRelaxationProblem(_BaseProblem):
    """Homogeneous linear relaxation problem with an exact Mittag-Leffler solution."""

    def __init__(
        self,
        *,
        alpha: float = 0.8,
        epsilon: float = 0.05,
        initial_condition: float = 1.0,
        horizon: float = 5.0,
    ) -> None:
        super().__init__(alpha=alpha, epsilon=epsilon, initial_condition=initial_condition, horizon=horizon)

    def residual(self, t: torch.Tensor, u: torch.Tensor, du_dt: torch.Tensor) -> torch.Tensor:
        eps_alpha = self._eps_alpha_tensor(t)
        return eps_alpha * du_dt + u

    def exact_solution(self, t: torch.Tensor) -> Optional[torch.Tensor]:
        eps_alpha = t.new_tensor(self._eps_alpha)
        clamped_t = torch.clamp(t, min=0.0)
        z = -torch.pow(clamped_t, self.alpha) / eps_alpha
        ml = mittag_leffler_approx(z, self.alpha, series_terms=12, switch_threshold=1.0)
        return self.initial_condition * ml


class VariableCoeffProblem(_BaseProblem):
    """Linear problem with time-varying reaction and external forcing."""

    def __init__(
        self,
        *,
        alpha: float = 0.8,
        epsilon: float = 0.05,
        initial_condition: float = 0.5,
        horizon: float = 5.0,
    ) -> None:
        super().__init__(alpha=alpha, epsilon=epsilon, initial_condition=initial_condition, horizon=horizon)
        self._freq = 4.0 * math.pi

    def residual(self, t: torch.Tensor, u: torch.Tensor, du_dt: torch.Tensor) -> torch.Tensor:
        eps_alpha = self._eps_alpha_tensor(t)
        coeff = 1.0 + 0.5 * torch.sin(t * self._freq)
        forcing = torch.cos(t)
        return eps_alpha * du_dt + coeff * u - forcing


class NonlinearLogisticProblem(_BaseProblem):
    """Nonlinear logistic-type relaxation with memory effects."""

    def __init__(
        self,
        *,
        alpha: float = 0.8,
        epsilon: float = 0.05,
        initial_condition: float = 0.1,
        horizon: float = 5.0,
        reaction_rate: float = 5.0,
    ) -> None:
        super().__init__(alpha=alpha, epsilon=epsilon, initial_condition=initial_condition, horizon=horizon)
        self._reaction_rate = float(reaction_rate)

    def residual(self, t: torch.Tensor, u: torch.Tensor, du_dt: torch.Tensor) -> torch.Tensor:
        del t  # Unused, kept for uniform signature.
        eps_alpha = self._eps_alpha_tensor(u)
        reaction = self._reaction_rate * u * (1.0 - u)
        return eps_alpha * du_dt + reaction


__all__ = [
    "LinearRelaxationProblem",
    "VariableCoeffProblem",
    "NonlinearLogisticProblem",
]
