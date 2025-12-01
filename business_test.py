"""Business case study: fractional advertising goodwill modeling with AIF-PINN."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from aif_pinn import (
    AbstractSPFDE,
    AIFPINNModel,
    DataGenerator,
    FractionalDerivativeOperator,
    PINNSolver,
    solve_variable_coeff,
)


class AdvertisingGoodwillProblem(AbstractSPFDE):
    """Fractional advertising goodwill dynamics with a seasonal driver."""

    def __init__(
        self,
        *,
        alpha: float,
        epsilon: float,
        horizon: float,
        initial_condition: float = 0.0,
        seasonal_amplitude: float = 0.3,
        seasonal_frequency: float = 2.0 * math.pi,
        campaign_level: float = 1.0,
    ) -> None:
        super().__init__(
            alpha=alpha,
            epsilon=epsilon,
            initial_condition=initial_condition,
            horizon=horizon,
        )
        self.seasonal_amplitude = float(seasonal_amplitude)
        self.seasonal_frequency = float(seasonal_frequency)
        self.campaign_level = float(campaign_level)

    def _eps_alpha_tensor(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_tensor(self._eps_alpha)

    def residual(self, t: torch.Tensor, u: torch.Tensor, du_dt: torch.Tensor) -> torch.Tensor:
        eps_alpha = self._eps_alpha_tensor(t)
        coeff = self.reaction_coefficient(t)
        forcing = self.forcing(t)
        return eps_alpha * du_dt + coeff * u - forcing

    def reaction_coefficient(self, t):
        freq = self.seasonal_frequency
        amp = self.seasonal_amplitude
        if torch.is_tensor(t):
            return 1.0 + amp * torch.cos(freq * t)
        array = np.asarray(t, dtype=np.float64)
        return 1.0 + amp * np.cos(freq * array)

    def forcing(self, t):
        if torch.is_tensor(t):
            return t.new_full(t.shape, self.campaign_level)
        array = np.asarray(t, dtype=np.float64)
        return np.full_like(array, self.campaign_level, dtype=np.float64)

    def exact_solution(self, t: torch.Tensor) -> Optional[torch.Tensor]:
        del t
        return None


@dataclass
class ForwardResult:
    pinn_error: float
    fdm_error: float
    figure_path: Path


@dataclass
class InverseResult:
    alpha_estimate: float
    epsilon_estimate: float
    alpha_true: float
    epsilon_true: float
    figure_path: Path


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_problem(alpha: float, epsilon: float, horizon: float) -> AdvertisingGoodwillProblem:
    return AdvertisingGoodwillProblem(
        alpha=alpha,
        epsilon=epsilon,
        horizon=horizon,
        initial_condition=0.0,
        seasonal_amplitude=0.3,
        seasonal_frequency=2.0 * math.pi,
        campaign_level=1.0,
    )


def train_solver(
    problem: AbstractSPFDE,
    *,
    num_collocation: int,
    device: torch.device,
    hidden_layers: Tuple[int, ...],
    adam_steps: int,
    lbfgs_steps: int,
    adam_lr: float,
    verbose: bool = False,
) -> PINNSolver:
    generator = DataGenerator(problem)
    grid = generator.generate_grid(num_collocation)
    operator = FractionalDerivativeOperator(
        grid,
        problem.alpha,
        dtype=torch.float32,
        device=device,
    )
    model = AIFPINNModel(
        alpha=problem.alpha,
        epsilon=problem.epsilon,
        initial_condition=problem.initial_condition,
        hidden_layers=hidden_layers,
        mittag_series_terms=12,
    )
    solver = PINNSolver(
        model,
        operator,
        problem,
        dtype=torch.float32,
        device=device,
    )
    solver.train(
        adam_steps=adam_steps,
        adam_lr=adam_lr,
        lbfgs_max_iter=lbfgs_steps,
        verbose=verbose,
    )
    return solver


def evaluate_model(model: torch.nn.Module, t_eval: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t_tensor = torch.as_tensor(t_eval, dtype=torch.float32, device=device).reshape(-1, 1)
        preds = model(t_tensor)
    return preds.cpu().numpy().reshape(-1)


def compute_reference_curve(
    alpha: float,
    epsilon: float,
    horizon: float,
    *,
    device: torch.device,
    num_collocation: int,
    adam_steps: int,
    lbfgs_steps: int,
    t_eval: np.ndarray,
) -> np.ndarray:
    ref_problem = build_problem(alpha, epsilon, horizon)
    solver = train_solver(
        ref_problem,
        num_collocation=num_collocation,
        device=device,
        hidden_layers=(256, 256, 128),
        adam_steps=adam_steps,
        lbfgs_steps=lbfgs_steps,
        adam_lr=5e-4,
        verbose=False,
    )
    return evaluate_model(solver.model, t_eval, device)


def run_forward_analysis(args: argparse.Namespace, device: torch.device) -> ForwardResult:
    problem = build_problem(alpha=0.75, epsilon=0.05, horizon=args.horizon)
    solver = train_solver(
        problem,
        num_collocation=args.forward_points,
        device=device,
        hidden_layers=(128, 128, 64),
        adam_steps=args.forward_adam,
        lbfgs_steps=args.forward_lbfgs,
        adam_lr=1e-3,
        verbose=args.verbose,
    )

    dense_t = np.linspace(0.0, problem.horizon, num=args.forward_dense_points, dtype=np.float32)
    pinn_curve = evaluate_model(solver.model, dense_t, device)

    fdm_grid, fdm_solution = solve_variable_coeff(problem, args.forward_points)
    fdm_interp = np.interp(dense_t, fdm_grid, fdm_solution)

    reference_curve = compute_reference_curve(
        problem.alpha,
        problem.epsilon,
        problem.horizon,
        device=device,
        num_collocation=args.reference_collocation,
        adam_steps=args.reference_adam,
        lbfgs_steps=args.reference_lbfgs,
        t_eval=dense_t,
    )

    fdm_mae = float(np.mean(np.abs(fdm_interp - reference_curve)))
    pinn_mae = float(np.mean(np.abs(pinn_curve - reference_curve)))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dense_t, reference_curve, label="Reference PINN (2000 pts)", color="#2ca02c", linewidth=2.0)
    ax.plot(dense_t, pinn_curve, label="AIF-PINN (100 pts)", color="#1f77b4", linewidth=2.2)
    ax.plot(dense_t, fdm_interp, label="FDM (100 pts)", color="#d62728", linestyle="--", linewidth=1.5)
    ax.scatter(
        fdm_grid,
        fdm_solution,
        label="FDM nodes",
        color="#ff9896",
        s=15,
        alpha=0.7,
        zorder=3,
    )
    ax.set_xlabel("Time t")
    ax.set_ylabel("Goodwill u(t)")
    ax.set_title("Fractional Advertising Goodwill: Forward analysis")
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(loc="best")
    fig.tight_layout()
    args.forward_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.forward_figure, dpi=300)
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    print(
        f"[Forward] PINN MAE vs reference: {pinn_mae:.4e} | "
        f"FDM MAE vs reference: {fdm_mae:.4e} "
        f"(figure saved to {args.forward_figure})"
    )
    return ForwardResult(pinn_error=pinn_mae, fdm_error=fdm_mae, figure_path=args.forward_figure)


def fractional_derivative(u: torch.Tensor, grid: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    alpha_scalar = alpha.reshape(())
    dtype = grid.dtype
    device = grid.device
    grid = grid.reshape(-1)
    n_points = grid.numel()
    if n_points < 2:
        return torch.zeros_like(u)

    inv_gamma = torch.exp(-torch.lgamma(torch.tensor(2.0, dtype=dtype, device=device) - alpha_scalar))
    deltas = torch.clamp(grid[1:] - grid[:-1], min=1e-12)
    power = 1.0 - alpha_scalar

    grid_prev = grid[:-1]
    grid_curr = grid[1:]
    grid_all = grid.unsqueeze(1)
    diff_prev = torch.clamp(grid_all - grid_prev.unsqueeze(0), min=1e-12)
    diff_curr = torch.clamp(grid_all - grid_curr.unsqueeze(0), min=1e-12)
    weights = (diff_prev.pow(power) - diff_curr.pow(power)) / deltas.unsqueeze(0)

    k_indices = torch.arange(1, n_points, device=device).unsqueeze(0)
    j_indices = torch.arange(n_points, device=device).unsqueeze(1)
    lower_mask = (j_indices >= k_indices).to(weights.dtype)
    weights = weights * lower_mask

    du = (u[1:] - u[:-1]).reshape(n_points - 1, -1)
    frac = inv_gamma * (weights @ du)
    frac = frac.reshape(u.shape[0], *u.shape[1:])
    return frac


def log_sample_times(horizon: float, num_points: int, device: torch.device) -> torch.Tensor:
    positive_points = max(num_points - 1, 1)
    t_positive = torch.logspace(
        math.log10(1e-4),
        math.log10(horizon),
        steps=positive_points,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)
    t_zero = torch.zeros(1, 1, device=device)
    return torch.cat([t_zero, t_positive], dim=0)


def train_inverse_model(
    model: AIFPINNModel,
    *,
    problem: AdvertisingGoodwillProblem,
    t_data: torch.Tensor,
    noisy_data: torch.Tensor,
    t_collocation: torch.Tensor,
    train_steps: int,
    lr: float,
    physics_weight: float,
    log_every: int,
) -> Dict[str, List[float]]:
    def _set_inverse_grad(enabled: bool) -> None:
        if hasattr(model, "logit_alpha"):
            model.logit_alpha.requires_grad = enabled
        if hasattr(model, "log_epsilon"):
            model.log_epsilon.requires_grad = enabled

    def _make_optimizer() -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ExponentialLR]:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        return optimizer, scheduler

    history: Dict[str, List[float]] = {"loss": [], "alpha": [], "epsilon": [], "grad_norm": []}
    grid_flat = t_collocation.squeeze(-1)

    warmup_steps = max(train_steps // 2, 1)
    discovery_steps = max(train_steps - warmup_steps, 1)
    phases = [
        {"name": "Warm-up", "steps": warmup_steps, "physics_weight": 0.0, "train_inverse": False},
        {"name": "Discovery", "steps": discovery_steps, "physics_weight": physics_weight, "train_inverse": True},
    ]

    global_step = 0
    for phase in phases:
        optimizer, scheduler = _make_optimizer()
        _set_inverse_grad(phase["train_inverse"])
        current_weight = phase["physics_weight"]

        for _ in range(phase["steps"]):
            optimizer.zero_grad()
            pred = model(t_data)
            data_loss = torch.mean(torch.square(pred - noisy_data))

            if current_weight > 0.0:
                u_collocation = model(t_collocation)
                alpha_tensor = model.alpha_tensor(device=t_collocation.device, dtype=t_collocation.dtype)
                epsilon_tensor = model.epsilon_tensor(device=t_collocation.device, dtype=t_collocation.dtype)
                frac_u = fractional_derivative(u_collocation, grid_flat, alpha_tensor)
                eps_alpha = torch.pow(epsilon_tensor, alpha_tensor)
                coeff = problem.reaction_coefficient(t_collocation)
                forcing = problem.forcing(t_collocation)
                residual = eps_alpha * frac_u + coeff * u_collocation - forcing
                physics_loss = torch.mean(torch.square(residual))
            else:
                physics_loss = torch.zeros(1, device=t_collocation.device, dtype=t_collocation.dtype)

            loss = data_loss + current_weight * physics_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if (global_step % log_every == 0) or (global_step == train_steps):
                print(
                    f"[Inverse] Step {global_step:04d}/{train_steps} "
                    f"Phase={phase['name']} | Loss={float(loss.detach().cpu().item()):.3e} | "
                    f"alpha={model.alpha_value():.4f} | epsilon={model.epsilon_value():.5f}"
                )

            history["loss"].append(float(loss.detach().cpu().item()))
            history["alpha"].append(model.alpha_value())
            history["epsilon"].append(model.epsilon_value())
            history["grad_norm"].append(float(grad_norm.detach().cpu().item()))

    _set_inverse_grad(True)

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=100,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad()
        pred = model(t_data)
        data_loss = torch.mean(torch.square(pred - noisy_data))

        u_collocation = model(t_collocation)
        alpha_tensor = model.alpha_tensor(device=t_collocation.device, dtype=t_collocation.dtype)
        epsilon_tensor = model.epsilon_tensor(device=t_collocation.device, dtype=t_collocation.dtype)
        frac_u = fractional_derivative(u_collocation, grid_flat, alpha_tensor)
        eps_alpha = torch.pow(epsilon_tensor, alpha_tensor)
        coeff = problem.reaction_coefficient(t_collocation)
        forcing = problem.forcing(t_collocation)
        residual = eps_alpha * frac_u + coeff * u_collocation - forcing
        physics_loss = torch.mean(torch.square(residual))

        loss = data_loss + physics_weight * physics_loss
        loss.backward()
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        pred = model(t_data)
        data_loss = torch.mean(torch.square(pred - noisy_data))

        u_collocation = model(t_collocation)
        alpha_tensor = model.alpha_tensor(device=t_collocation.device, dtype=t_collocation.dtype)
        epsilon_tensor = model.epsilon_tensor(device=t_collocation.device, dtype=t_collocation.dtype)
        frac_u = fractional_derivative(u_collocation, grid_flat, alpha_tensor)
        eps_alpha = torch.pow(epsilon_tensor, alpha_tensor)
        coeff = problem.reaction_coefficient(t_collocation)
        forcing = problem.forcing(t_collocation)
        residual = eps_alpha * frac_u + coeff * u_collocation - forcing
        physics_loss = torch.mean(torch.square(residual))

        final_loss = float((data_loss + physics_weight * physics_loss).detach().cpu().item())
    print(f"[Inverse] L-BFGS complete | Loss={final_loss:.3e}")

    return history


def run_market_analysis(args: argparse.Namespace, device: torch.device) -> InverseResult:
    alpha_true = 0.75
    epsilon_true = 0.05
    truth_problem = build_problem(alpha=alpha_true, epsilon=epsilon_true, horizon=args.horizon)

    dense_t = np.linspace(0.0, truth_problem.horizon, num=args.forward_dense_points, dtype=np.float32)
    print("[Inverse] Building high-resolution reference curve for synthetic sales data...")
    reference_curve = compute_reference_curve(
        alpha_true,
        epsilon_true,
        truth_problem.horizon,
        device=device,
        num_collocation=args.reference_collocation,
        adam_steps=args.reference_adam,
        lbfgs_steps=args.reference_lbfgs,
        t_eval=dense_t,
    )

    t_data = log_sample_times(truth_problem.horizon, args.inverse_data_points, device)
    with torch.no_grad():
        reference_tensor = torch.as_tensor(
            np.interp(t_data.cpu().numpy().reshape(-1), dense_t, reference_curve),
            dtype=torch.float32,
            device=device,
        ).reshape(-1, 1)
    noise_std = args.inverse_noise * torch.std(reference_tensor)
    noisy_data = reference_tensor + noise_std * torch.randn_like(reference_tensor)

    generator = DataGenerator(truth_problem)
    collocation_grid = generator.generate_grid(args.inverse_collocation)
    t_collocation = torch.as_tensor(collocation_grid, dtype=torch.float32, device=device).reshape(-1, 1)

    model = AIFPINNModel(
        alpha=args.inverse_alpha_init,
        epsilon=args.inverse_epsilon_init,
        initial_condition=truth_problem.initial_condition,
        hidden_layers=(128, 128, 64),
        mittag_series_terms=10,
        inverse_problem=True,
    ).to(device)

    print("[Inverse] Starting two-phase training + L-BFGS refinement...")
    train_inverse_model(
        model,
        problem=truth_problem,
        t_data=t_data,
        noisy_data=noisy_data,
        t_collocation=t_collocation,
        train_steps=args.inverse_adam,
        lr=args.inverse_lr,
        physics_weight=args.physics_weight,
        log_every=args.log_every,
    )

    dense_tensor = torch.as_tensor(dense_t, dtype=torch.float32, device=device).reshape(-1, 1)
    with torch.no_grad():
        recovered_curve = model(dense_tensor).cpu().numpy().reshape(-1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dense_t, reference_curve, label="Ground truth (reference PINN)", color="#2ca02c", linewidth=2.0)
    ax.plot(dense_t, recovered_curve, label="Recovered curve", color="#1f77b4", linewidth=2.2)
    ax.scatter(
        t_data.cpu().numpy(),
        noisy_data.cpu().numpy(),
        label="Noisy sales data",
        color="#ff7f0e",
        s=25,
        alpha=0.75,
    )
    ax.set_xlabel("Time t")
    ax.set_ylabel("Observed goodwill / sales proxy")
    ax.set_title("Market analytics: recovering memory and forgetting rate")
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(loc="best")
    fig.tight_layout()
    args.inverse_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.inverse_figure, dpi=300)
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    alpha_est = model.alpha_value()
    epsilon_est = model.epsilon_value()
    print(
        "[Inverse] Parameter recovery:\n"
        f"  alpha  true={alpha_true:.2f}, recovered={alpha_est:.4f}\n"
        f"  epsilon true={epsilon_true:.2f}, recovered={epsilon_est:.5f}\n"
        f"Figure saved to {args.inverse_figure}"
    )
    return InverseResult(
        alpha_estimate=alpha_est,
        epsilon_estimate=epsilon_est,
        alpha_true=alpha_true,
        epsilon_true=epsilon_true,
        figure_path=args.inverse_figure,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advertising goodwill scenario solved with AIF-PINN (forward + inverse analyses)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("forward", "inverse", "all"),
        default="all",
        help="Which scenario to execute.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--horizon", type=float, default=3.0, help="Business horizon in years.")
    parser.add_argument("--no-show", action="store_true", help="Do not display matplotlib windows.")
    parser.add_argument("--verbose", action="store_true", help="Print optimizer progress during forward solve.")

    parser.add_argument("--forward-points", type=int, default=100, help="Number of collocation/grid points for forward run.")
    parser.add_argument("--forward-dense-points", type=int, default=600, help="Dense evaluation grid for plotting.")
    parser.add_argument("--forward-adam", type=int, default=2000, help="Adam iterations for forward PINN.")
    parser.add_argument("--forward-lbfgs", type=int, default=800, help="L-BFGS iterations for forward PINN.")
    parser.add_argument("--forward-figure", type=Path, default=Path("figures") / "business_forward.png", help="Forward plot path.")

    parser.add_argument("--reference-collocation", type=int, default=2000, help="High-accuracy reference collocation points.")
    parser.add_argument("--reference-adam", type=int, default=3000, help="Adam steps for reference PINN.")
    parser.add_argument("--reference-lbfgs", type=int, default=1500, help="L-BFGS steps for reference PINN.")

    parser.add_argument("--inverse-data-points", type=int, default=60, help="Number of noisy observations.")
    parser.add_argument("--inverse-collocation", type=int, default=256, help="Collocation size for inverse problem.")
    parser.add_argument("--inverse-adam", type=int, default=2500, help="Total Adam iterations for inverse training.")
    parser.add_argument("--inverse-lr", type=float, default=1e-2, help="Learning rate for Adam (inverse).")
    parser.add_argument("--inverse-noise", type=float, default=0.05, help="Relative noise level for synthetic sales.")
    parser.add_argument("--inverse-alpha-init", type=float, default=0.5, help="Initial guess for alpha.")
    parser.add_argument("--inverse-epsilon-init", type=float, default=0.1, help="Initial guess for epsilon.")
    parser.add_argument("--physics-weight", type=float, default=1e-3, help="Weight of physics loss during discovery.")
    parser.add_argument("--log-every", type=int, default=200, help="Logging cadence for inverse training.")
    parser.add_argument(
        "--inverse-figure",
        type=Path,
        default=Path("figures") / "business_inverse.png",
        help="Inverse problem plot path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    forward_result: Optional[ForwardResult] = None
    inverse_result: Optional[InverseResult] = None

    if args.mode in {"forward", "all"}:
        forward_result = run_forward_analysis(args, device)
    if args.mode in {"inverse", "all"}:
        inverse_result = run_market_analysis(args, device)

    if forward_result is not None:
        print(
            f"[Summary] Forward study stored at {forward_result.figure_path}. "
            f"MAE (PINN)={forward_result.pinn_error:.3e}, MAE (FDM)={forward_result.fdm_error:.3e}"
        )
    if inverse_result is not None:
        alpha_err = abs(inverse_result.alpha_estimate - inverse_result.alpha_true) / inverse_result.alpha_true * 100.0
        eps_err = (
            abs(inverse_result.epsilon_estimate - inverse_result.epsilon_true) / inverse_result.epsilon_true * 100.0
        )
        print(
            f"[Summary] Inverse study stored at {inverse_result.figure_path}. "
            f"alpha error={alpha_err:.2f}% | epsilon error={eps_err:.2f}%"
        )


if __name__ == "__main__":
    main()
