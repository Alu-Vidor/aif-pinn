"""Problem abstractions shared by AIF-PINN components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from torch import Tensor


class AbstractSPFDE(ABC):
    """Abstract base class describing a single-state fractional differential equation."""

    def __init__(
        self,
        *,
        alpha: float,
        epsilon: float,
        initial_condition: float,
        horizon: float,
    ) -> None:
        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        if not (0.0 < epsilon):
            raise ValueError("epsilon must be positive.")
        if horizon <= 0.0:
            raise ValueError("horizon must be positive.")

        self._alpha = float(alpha)
        self._epsilon = float(epsilon)
        self._initial_condition = float(initial_condition)
        self._horizon = float(horizon)
        self._eps_alpha = float(pow(self._epsilon, self._alpha))

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def initial_condition(self) -> float:
        return self._initial_condition

    @property
    def horizon(self) -> float:
        return self._horizon

    def boundary_layer_width(self) -> float:
        """Width of the inner boundary layer, defaults to a multiple of epsilon."""

        return 5.0 * self._epsilon

    @abstractmethod
    def residual(self, t: "Tensor", u: "Tensor", du_dt: "Tensor") -> "Tensor":
        """Residual F(t, u, D^alpha u) that the solver drives to zero."""

    def exact_solution(self, t: "Tensor") -> Optional["Tensor"]:
        """Optional reference solution for visual comparison."""

        return None


__all__ = ["AbstractSPFDE"]
