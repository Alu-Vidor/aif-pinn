"""Neural network architectures used in the AIF-PINN solver."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from .mittag import mittag_leffler_approx


class AIFPINNModel(nn.Module):
    """PINN model with asymptotic embeddings and a hard boundary condition constraint."""

    def __init__(
        self,
        *,
        alpha: float,
        epsilon: float,
        initial_condition: float,
        hidden_layers: Sequence[int] = (64, 64),
        activation: type[nn.Module] = nn.Tanh,
        mittag_series_terms: int = 6,
        mittag_switch_threshold: float = 1.0,
        inverse_problem: bool = False,
    ) -> None:
        super().__init__()
        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if mittag_series_terms <= 0:
            raise ValueError("mittag_series_terms must be positive.")

        self.alpha = float(alpha)
        self.hidden_layers = tuple(int(h) for h in hidden_layers)
        self.activation_cls = activation
        self.mittag_series_terms = mittag_series_terms
        self.mittag_switch_threshold = mittag_switch_threshold
        self.inverse_problem = bool(inverse_problem)

        if self.inverse_problem:
            alpha_tensor = torch.tensor(float(alpha))
            alpha_tensor = torch.clamp(alpha_tensor, 1e-3, 1.0 - 1e-3)
            self.logit_alpha = nn.Parameter(torch.logit(alpha_tensor))
            self.log_epsilon = nn.Parameter(torch.log(torch.tensor(float(epsilon))))
            self._epsilon = None
        else:
            self.register_buffer("_epsilon", torch.tensor(float(epsilon)))

        self.register_buffer("_u0", torch.tensor(float(initial_condition)))

        self.net = self._build_network(input_dim=4)
        self._eps_floor = 1e-6

    def _build_network(self, input_dim: int) -> nn.Module:
        layers: list[nn.Module] = []
        prev_dim = input_dim
        if not self.hidden_layers:
            layers.append(nn.Linear(prev_dim, 1))
            return nn.Sequential(*layers)

        for width in self.hidden_layers:
            if width <= 0:
                raise ValueError("hidden layer sizes must be positive integers.")
            layers.append(nn.Linear(prev_dim, width))
            layers.append(self.activation_cls())
            prev_dim = width

        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def _resolved_alpha(self, ref: torch.Tensor) -> torch.Tensor:
        if self.inverse_problem:
            alpha = torch.sigmoid(self.logit_alpha)
            return alpha.to(dtype=ref.dtype, device=ref.device)
        return ref.new_tensor(self.alpha)

    def _resolved_epsilon(self, ref: torch.Tensor) -> torch.Tensor:
        if self.inverse_problem:
            eps = torch.exp(self.log_epsilon) + self._eps_floor
            return eps.to(dtype=ref.dtype, device=ref.device)
        if self._epsilon is None:
            raise RuntimeError("epsilon buffer is not initialized.")
        return self._epsilon.to(dtype=ref.dtype, device=ref.device)

    def _feature_embedding(self, x: torch.Tensor) -> torch.Tensor:
        eps = self._resolved_epsilon(x)
        alpha = self._resolved_alpha(x)
        eps_alpha = torch.pow(eps, alpha)
        x_scaled = x / eps
        x_clamped = torch.clamp(x, min=0.0)
        x_alpha = torch.pow(torch.clamp(x_clamped, min=1e-12), alpha)
        z = -x_alpha / eps_alpha
        mittag = mittag_leffler_approx(
            z,
            float(alpha.detach().cpu().item()) if torch.is_tensor(alpha) else self.alpha,
            series_terms=self.mittag_series_terms,
            switch_threshold=self.mittag_switch_threshold,
        )
        return torch.cat([x, x_scaled, x_alpha, mittag], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if x.size(-1) != 1:
            raise ValueError("Input tensor must be of shape (N, 1).")

        features = self._feature_embedding(x)
        raw_output = self.net(features)

        eps = self._resolved_epsilon(x)
        u0 = self._u0.to(dtype=x.dtype, device=x.device)
        constraint = 1.0 - torch.exp(-x / eps)

        return u0 + constraint * raw_output

    def alpha_tensor(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        ref = torch.zeros(1, 1, device=device or self._u0.device, dtype=dtype or self._u0.dtype)
        return self._resolved_alpha(ref)

    def epsilon_tensor(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        ref = torch.zeros(1, 1, device=device or self._u0.device, dtype=dtype or self._u0.dtype)
        return self._resolved_epsilon(ref)

    def alpha_value(self) -> float:
        if self.inverse_problem:
            return float(torch.sigmoid(self.logit_alpha).detach().cpu().item())
        return self.alpha

    def epsilon_value(self) -> float:
        if self.inverse_problem:
            return float((torch.exp(self.log_epsilon) + self._eps_floor).detach().cpu().item())
        if self._epsilon is None:
            raise RuntimeError("epsilon buffer is not initialized.")
        return float(self._epsilon.detach().cpu().item())


__all__ = ["AIFPINNModel"]
