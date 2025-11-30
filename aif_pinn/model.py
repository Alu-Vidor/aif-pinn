"""Neural network architectures used in the AIF-PINN solver."""

from __future__ import annotations

from typing import Sequence

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

        self.register_buffer("_epsilon", torch.tensor(float(epsilon)))
        self.register_buffer("_u0", torch.tensor(float(initial_condition)))

        self.net = self._build_network(input_dim=4)

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

    def _feature_embedding(self, x: torch.Tensor) -> torch.Tensor:
        eps = self._epsilon.to(dtype=x.dtype, device=x.device)
        eps_alpha = torch.pow(eps, self.alpha)
        x_scaled = x / eps
        x_clamped = torch.clamp(x, min=0.0)
        x_alpha = torch.pow(x_clamped, self.alpha)
        z = -x_alpha / eps_alpha
        mittag = mittag_leffler_approx(
            z,
            self.alpha,
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

        eps = self._epsilon.to(dtype=x.dtype, device=x.device)
        u0 = self._u0.to(dtype=x.dtype, device=x.device)
        constraint = 1.0 - torch.exp(-x / eps)

        return u0 + constraint * raw_output


__all__ = ["AIFPINNModel"]
