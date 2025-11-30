"""Inverse problem demonstration with AIF-PINN discovering alpha and epsilon."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from aif_pinn import AIFPINNModel, DataGenerator, LinearRelaxationProblem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover alpha and epsilon from noisy observations via AIF-PINN.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num-observations", type=int, default=50, help="Number of noisy data points.")
    parser.add_argument("--num-collocation", type=int, default=256, help="Number of collocation points for physics loss.")
    parser.add_argument("--noise-level", type=float, default=0.05, help="Relative noise level (fraction of std).")
    parser.add_argument("--train-steps", type=int, default=2000, help="Number of Adam iterations.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate.")
    parser.add_argument("--physics-weight", type=float, default=1e-3, help="Weight for the physics residual.")
    parser.add_argument("--log-every", type=int, default=200, help="Logging frequency for training metrics.")
    parser.add_argument(
        "--figure-solution",
        type=Path,
        default=Path("figures") / "inverse_solution.png",
        help="Where to save the reconstructed solution plot.",
    )
    parser.add_argument(
        "--figure-params",
        type=Path,
        default=Path("figures") / "inverse_parameters.png",
        help="Where to save the parameter convergence plot.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display matplotlib windows.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def generate_noisy_data(
    problem: LinearRelaxationProblem,
    *,
    num_points: int,
    noise_level: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_points <= 0:
        raise ValueError("num_points must be positive.")

    positive_points = max(num_points - 1, 1)
    t_positive = torch.logspace(
        math.log10(1e-4),
        math.log10(problem.horizon),
        steps=positive_points,
        device=device,
        dtype=torch.float32,
    ).reshape(-1, 1)
    t_zero = torch.zeros(1, 1, device=device)
    t = torch.cat([t_zero, t_positive], dim=0)
    with torch.no_grad():
        clean = problem.exact_solution(t).to(device)
    noise_std = noise_level * torch.std(clean)
    noisy = clean + noise_std * torch.randn_like(clean)
    return t, clean.detach(), noisy.detach()


def build_collocation_tensor(problem: LinearRelaxationProblem, num_points: int, device: torch.device) -> torch.Tensor:
    generator = DataGenerator(problem)
    grid = generator.generate_grid(num_points)
    return torch.as_tensor(grid, dtype=torch.float32, device=device).reshape(-1, 1)


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


def train_inverse_model(
    model: AIFPINNModel,
    *,
    t_data: torch.Tensor,
    noisy_data: torch.Tensor,
    t_collocation: torch.Tensor,
    train_steps: int,
    lr: float,
    physics_weight: float,
    log_every: int,
) -> Dict[str, List[float]]:
    def _set_inverse_params_grad(enabled: bool) -> None:
        if not getattr(model, "inverse_problem", False):
            return
        if hasattr(model, "logit_alpha"):
            model.logit_alpha.requires_grad = enabled
        if hasattr(model, "log_epsilon"):
            model.log_epsilon.requires_grad = enabled

    def _make_optimizer() -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ExponentialLR]:
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters available for optimization.")
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        return optimizer, scheduler

    history: Dict[str, List[float]] = {"loss": [], "alpha": [], "epsilon": [], "grad_norm": []}
    grid_flat = t_collocation.squeeze(-1)

    if train_steps > 0:
        warmup_steps = max(1, train_steps // 2)
        discovery_steps = train_steps - warmup_steps
        phases = [
            {"name": "Warm-up", "steps": warmup_steps, "physics_weight": 0.0, "train_inverse": False},
            {"name": "Discovery", "steps": discovery_steps, "physics_weight": physics_weight, "train_inverse": True},
        ]
    else:
        phases = []

    global_step = 0
    for phase in phases:
        phase_steps = phase["steps"]
        if phase_steps <= 0:
            continue

        _set_inverse_params_grad(phase["train_inverse"])
        optimizer, scheduler = _make_optimizer()
        current_weight = phase["physics_weight"]

        for _ in range(phase_steps):
            optimizer.zero_grad()
            pred_data = model(t_data)
            data_loss = torch.mean(torch.square(pred_data - noisy_data))

            if current_weight > 0.0:
                u_collocation = model(t_collocation)
                alpha_tensor = model.alpha_tensor(device=t_collocation.device, dtype=t_collocation.dtype)
                epsilon_tensor = model.epsilon_tensor(device=t_collocation.device, dtype=t_collocation.dtype)
                frac_u = fractional_derivative(u_collocation, grid_flat, alpha_tensor)
                eps_alpha = torch.pow(epsilon_tensor, alpha_tensor)
                residual = eps_alpha * frac_u + u_collocation
                physics_loss = torch.mean(torch.square(residual))
            else:
                physics_loss = torch.zeros(1, device=t_collocation.device, dtype=t_collocation.dtype)

            loss = data_loss + current_weight * physics_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            with torch.no_grad():
                history["loss"].append(float(loss.detach().cpu().item()))
                history["alpha"].append(model.alpha_value())
                history["epsilon"].append(model.epsilon_value())
                history["grad_norm"].append(float(grad_norm.detach().cpu().item()))

            if (global_step % log_every == 0) or (global_step == train_steps):
                print(
                    f"Step {global_step:5d}/{train_steps} | "
                    f"Phase={phase['name']} | "
                    f"Loss={history['loss'][-1]:.3e} | "
                    f"Data={float(data_loss.detach().cpu().item()):.3e} | "
                    f"Physics={float(physics_loss.detach().cpu().item()):.3e} | "
                    f"alpha={history['alpha'][-1]:.4f} | "
                    f"epsilon={history['epsilon'][-1]:.5f} | "
                    f"Grad={history['grad_norm'][-1]:.3e}"
                )

    _set_inverse_params_grad(True)

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=100,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad()
        pred_data = model(t_data)
        data_loss = torch.mean(torch.square(pred_data - noisy_data))

        u_collocation = model(t_collocation)
        alpha_tensor = model.alpha_tensor(device=t_collocation.device, dtype=t_collocation.dtype)
        epsilon_tensor = model.epsilon_tensor(device=t_collocation.device, dtype=t_collocation.dtype)
        frac_u = fractional_derivative(u_collocation, grid_flat, alpha_tensor)
        eps_alpha = torch.pow(epsilon_tensor, alpha_tensor)
        residual = eps_alpha * frac_u + u_collocation
        physics_loss = torch.mean(torch.square(residual))

        loss = data_loss + physics_weight * physics_loss
        loss.backward()
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        pred_data = model(t_data)
        data_loss = torch.mean(torch.square(pred_data - noisy_data))
        u_collocation = model(t_collocation)
        alpha_tensor = model.alpha_tensor(device=t_collocation.device, dtype=t_collocation.dtype)
        epsilon_tensor = model.epsilon_tensor(device=t_collocation.device, dtype=t_collocation.dtype)
        frac_u = fractional_derivative(u_collocation, grid_flat, alpha_tensor)
        eps_alpha = torch.pow(epsilon_tensor, alpha_tensor)
        residual = eps_alpha * frac_u + u_collocation
        physics_loss = torch.mean(torch.square(residual))
        loss = data_loss + physics_weight * physics_loss
        final_alpha = model.alpha_value()
        final_epsilon = model.epsilon_value()
        history["loss"].append(float(loss.detach().cpu().item()))
        history["alpha"].append(final_alpha)
        history["epsilon"].append(final_epsilon)
        print(
            "L-BFGS Done | "
            f"Loss={float(loss.detach().cpu().item()):.3e} | "
            f"Data={float(data_loss.detach().cpu().item()):.3e} | "
            f"Physics={float(physics_loss.detach().cpu().item()):.3e} | "
            f"alpha={final_alpha:.4f} | "
            f"epsilon={final_epsilon:.5f}"
        )

    return history


def plot_solution(
    t_dense: torch.Tensor,
    true_solution: torch.Tensor,
    observed_t: torch.Tensor,
    noisy_data: torch.Tensor,
    prediction: torch.Tensor,
    *,
    figure_path: Path,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t_dense.cpu().numpy(), true_solution.cpu().numpy(), label="Ground Truth", color="#2ca02c", linewidth=2.0)
    ax.scatter(
        observed_t.cpu().numpy(),
        noisy_data.cpu().numpy(),
        label="Noisy observations",
        color="#ff7f0e",
        s=20,
        alpha=0.7,
    )
    ax.plot(
        t_dense.cpu().numpy(),
        prediction.cpu().numpy(),
        label="AIF-PINN",
        color="#1f77b4",
        linewidth=2.2,
    )
    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    ax.set_title("Inverse problem: reconstruction of u(t)")
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(loc="best")
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_parameters(
    alpha_history: List[float],
    epsilon_history: List[float],
    *,
    alpha_true: float,
    epsilon_true: float,
    figure_path: Path,
    show: bool,
) -> None:
    steps = np.arange(1, len(alpha_history) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, alpha_history, label="alpha (learned)", color="#1f77b4", linewidth=2.0)
    ax.axhline(alpha_true, color="#1f77b4", linestyle="--", linewidth=1.2, label="alpha true")
    ax.plot(steps, epsilon_history, label="epsilon (learned)", color="#d62728", linewidth=2.0)
    ax.axhline(epsilon_true, color="#d62728", linestyle="--", linewidth=1.2, label="epsilon true")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Parameter value")
    ax.set_title("Convergence of recovered parameters")
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(loc="best")
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    alpha_true = 0.8
    epsilon_true = 0.05
    problem = LinearRelaxationProblem(alpha=alpha_true, epsilon=epsilon_true, initial_condition=1.0, horizon=5.0)

    print("Генерация данных...")
    
    t_data, _, noisy_data = generate_noisy_data(
        problem,
        num_points=args.num_observations,
        noise_level=args.noise_level,
        device=device,
    )
    t_collocation = build_collocation_tensor(problem, args.num_collocation, device)

    print("Данные сгенерированы.")

    model = AIFPINNModel(
        alpha=0.5,
        epsilon=0.1,
        initial_condition=problem.initial_condition,
        hidden_layers=(128, 128, 64),
        mittag_series_terms=8,
        inverse_problem=True,
    ).to(device)

    history = train_inverse_model(
        model,
        t_data=t_data,
        noisy_data=noisy_data,
        t_collocation=t_collocation,
        train_steps=args.train_steps,
        lr=args.lr,
        physics_weight=args.physics_weight,
        log_every=args.log_every,
    )

    t_dense = torch.linspace(0.0, problem.horizon, steps=400, dtype=torch.float32, device=device).reshape(-1, 1)
    with torch.no_grad():
        pinn_prediction = model(t_dense)
        true_curve = problem.exact_solution(t_dense).to(device)

    plot_solution(
        t_dense,
        true_curve,
        t_data,
        noisy_data,
        pinn_prediction,
        figure_path=args.figure_solution,
        show=not args.no_show,
    )
    plot_parameters(
        history["alpha"],
        history["epsilon"],
        alpha_true=alpha_true,
        epsilon_true=epsilon_true,
        figure_path=args.figure_params,
        show=not args.no_show,
    )

    alpha_error = abs(model.alpha_value() - alpha_true) / alpha_true * 100.0
    epsilon_error = abs(model.epsilon_value() - epsilon_true) / epsilon_true * 100.0
    print("\nRecovered parameters:")
    print(f"  alpha  = {model.alpha_value():.4f} (error {alpha_error:.2f}%)")
    print(f"  epsilon= {model.epsilon_value():.5f} (error {epsilon_error:.2f}%)")


if __name__ == "__main__":
    main()
