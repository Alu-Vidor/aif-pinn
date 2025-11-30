from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    import jax.numpy as jnp
    import torch


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

    def rhs(self, x: "torch.Tensor") -> "torch.Tensor":
        """Right-hand side forcing term f(x); override to encode problem-specific data."""

        if not torch.is_tensor(x):
            raise TypeError("rhs expects a torch.Tensor input.")
        return torch.zeros_like(x)

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
        log_points = np.logspace(np.log10(positive_min), np.log10(layer_extent), num=n_layer, endpoint=False, dtype=np.float64)
        trimmed = log_points[: n_layer - 1]

        return np.concatenate(([0.0], trimmed))

    def _uniform_outer(self, layer_extent: float, n_outer: int) -> np.ndarray:
        """Generate uniform points in the outer domain, starting at the layer edge."""

        if n_outer <= 0:
            return np.empty(0, dtype=np.float64)

        if layer_extent >= self.config.horizon:
            return np.linspace(0.0, self.config.horizon, num=n_outer, endpoint=True, dtype=np.float64)

        return np.linspace(layer_extent, self.config.horizon, num=n_outer, endpoint=True, dtype=np.float64)


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
        self._matrix_cache = {}
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


def _mittag_leffler_approx(
    z: "torch.Tensor",
    alpha: float,
    *,
    series_terms: int = 6,
    switch_threshold: float = 1.0,
) -> "torch.Tensor":
    """Piecewise approximation of Mittag-Leffler using a Taylor series + 1/z asymptotic."""

    if series_terms <= 0:
        raise ValueError("series_terms must be positive.")
    if not torch.is_tensor(z):
        raise TypeError("z must be a torch.Tensor.")

    abs_z = torch.abs(z)
    result = torch.zeros_like(z)
    small_mask = abs_z <= switch_threshold

    if small_mask.any():
        # Use a truncated Taylor expansion around z=0 for smooth behaviour.
        small_z = z[small_mask]
        series = torch.zeros_like(small_z)
        denominators = [math.gamma(alpha * k + 1.0) for k in range(series_terms)]
        for k, denom in enumerate(denominators):
            series = series + small_z.pow(k) / denom
        result[small_mask] = series

    if (~small_mask).any():
        # Simple asymptotic 1/z behaviour in the outer region of the boundary layer.
        large_z = z[~small_mask]
        safe_large = torch.where(
            torch.abs(large_z) < 1e-8,
            torch.full_like(large_z, 1e-8),
            large_z,
        )
        result[~small_mask] = 1.0 / safe_large

    return result


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

    def _feature_embedding(self, x: "torch.Tensor") -> "torch.Tensor":
        eps = self._epsilon.to(dtype=x.dtype, device=x.device)
        eps_alpha = torch.pow(eps, self.alpha)
        x_scaled = x / eps
        x_clamped = torch.clamp(x, min=0.0)
        x_alpha = torch.pow(x_clamped, self.alpha)
        z = -x_alpha / eps_alpha
        mittag = _mittag_leffler_approx(
            z,
            self.alpha,
            series_terms=self.mittag_series_terms,
            switch_threshold=self.mittag_switch_threshold,
        )
        return torch.cat([x, x_scaled, x_alpha, mittag], dim=-1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
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


class PINNSolver:
    """Orchestrate training with a hybrid Adam + L-BFGS optimization schedule."""

    def __init__(
        self,
        model: AIFPINNModel,
        operator: FractionalDerivativeOperator,
        problem_config: ProblemConfig,
        *,
        rhs_function: Optional[Callable[["torch.Tensor"], "torch.Tensor"]] = None,
        dtype: Optional["torch.dtype"] = None,
        device: Optional["torch.device"] = None,
    ) -> None:
        if rhs_function is not None and not callable(rhs_function):
            raise TypeError("rhs_function must be callable.")

        self.model = model
        self.operator = operator
        self.problem_config = problem_config
        self._rhs_fn = rhs_function

        sample_param = next(self.model.parameters(), None)
        resolved_device = device or (sample_param.device if sample_param is not None else torch.device("cpu"))
        resolved_dtype = dtype or (sample_param.dtype if sample_param is not None else torch.float32)

        self.model.to(device=resolved_device, dtype=resolved_dtype)
        self.device = resolved_device
        self.dtype = resolved_dtype

        self.grid_tensor = torch.as_tensor(
            self.operator.grid,
            dtype=self.dtype,
            device=self.device,
        ).reshape(-1, 1)

        eps_tensor = torch.tensor(float(self.problem_config.epsilon), dtype=self.dtype, device=self.device)
        self._eps_alpha = torch.pow(eps_tensor, float(self.problem_config.alpha))
        self._lambda = torch.tensor(float(self.problem_config.lambda_coeff), dtype=self.dtype, device=self.device)

    def _rhs(self, x: "torch.Tensor") -> "torch.Tensor":
        if self._rhs_fn is not None:
            return self._rhs_fn(x)
        return self.problem_config.rhs(x)

    def loss_function(self) -> "torch.Tensor":
        """Residual loss of the SPFDE."""

        x = self.grid_tensor
        u_pred = self.model(x)
        Pu = self.operator.apply(u_pred)
        rhs = self._rhs(x)

        residual = self._eps_alpha * Pu + self._lambda * u_pred - rhs
        return torch.mean(torch.square(residual))

    def train(
        self,
        *,
        adam_steps: int = 2000,
        adam_lr: float = 1e-3,
        adam_weight_decay: float = 0.0,
        lbfgs_max_iter: int = 500,
        lbfgs_history_size: int = 50,
        lbfgs_tolerance_grad: float = 1e-9,
        lbfgs_tolerance_change: float = 1e-9,
        verbose: bool = False,
        log_every: int = 200,
    ) -> dict[str, list[float]]:
        """Train with Adam for coarse convergence, then refine with L-BFGS."""

        history: dict[str, list[float]] = {"adam": [], "lbfgs": []}
        self.model.train()

        if adam_steps > 0:
            adam = torch.optim.Adam(self.model.parameters(), lr=adam_lr, weight_decay=adam_weight_decay)
            for step in range(adam_steps):
                adam.zero_grad()
                loss = self.loss_function()
                loss.backward()
                adam.step()

                loss_value = float(loss.detach().cpu().item())
                history["adam"].append(loss_value)
                if verbose and ((step + 1) % log_every == 0 or step == adam_steps - 1):
                    print(f"[Adam] Step {step + 1}/{adam_steps} - Loss: {loss_value:.3e}")

        if lbfgs_max_iter > 0:
            lbfgs = torch.optim.LBFGS(
                self.model.parameters(),
                lr=1.0,
                max_iter=lbfgs_max_iter,
                history_size=lbfgs_history_size,
                line_search_fn="strong_wolfe",
                tolerance_grad=lbfgs_tolerance_grad,
                tolerance_change=lbfgs_tolerance_change,
            )

            def closure() -> "torch.Tensor":
                self.model.zero_grad()
                loss = self.loss_function()
                loss.backward()
                return loss

            lbfgs.step(closure)
            with torch.no_grad():
                final_loss = float(self.loss_function().cpu().item())
            history["lbfgs"].append(final_loss)
            if verbose:
                print(f"[L-BFGS] Final loss after {lbfgs_max_iter} iterations: {final_loss:.3e}")

        return history
