"""Training orchestration utilities for the AIF-PINN model."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .model import AIFPINNModel
from .operators import FractionalDerivativeOperator
from .problem import ProblemConfig


class PINNSolver:
    """Orchestrate training with a hybrid Adam + L-BFGS optimization schedule."""

    def __init__(
        self,
        model: AIFPINNModel,
        operator: FractionalDerivativeOperator,
        problem_config: ProblemConfig,
        *,
        rhs_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
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

    def _rhs(self, x: torch.Tensor) -> torch.Tensor:
        if self._rhs_fn is not None:
            return self._rhs_fn(x)
        return self.problem_config.rhs(x)

    def loss_function(self) -> torch.Tensor:
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

            def closure() -> torch.Tensor:
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


__all__ = ["PINNSolver"]
