"""Compare AIF-PINN with a uniform-grid FDM baseline on a stiff SPFDE."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from aif_pinn import (
    AIFPINNModel,
    DataGenerator,
    FDMSolver,
    FractionalDerivativeOperator,
    LinearRelaxationProblem,
    PINNSolver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize AIF-PINN vs FDM on the linear relaxation problem.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Fractional order alpha.")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Singular perturbation parameter epsilon.")
    parser.add_argument("--initial-condition", type=float, default=1.0, help="Initial value u(0).")
    parser.add_argument("--horizon", type=float, default=5.0, help="Time horizon.")
    parser.add_argument("--fdm-points", type=int, default=100, help="Number of uniform grid points for FDM.")
    parser.add_argument("--num-collocation", type=int, default=100, help="Number of collocation points for AIF-PINN.")
    parser.add_argument("--adam-steps", type=int, default=2000, help="Number of Adam iterations for AIF-PINN.")
    parser.add_argument("--lbfgs-iter", type=int, default=1500, help="L-BFGS refinement iterations for AIF-PINN.")
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--adam-weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument("--eval-points", type=int, default=2000, help="Dense evaluation points for plotting.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures") / "method_comparison.png",
        help="Path to save the comparison figure.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not show the matplotlib window.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_pinn_solver(problem: LinearRelaxationProblem, num_collocation: int, device: torch.device) -> PINNSolver:
    generator = DataGenerator(problem)
    grid = generator.generate_grid(num_collocation)
    operator = FractionalDerivativeOperator(grid, problem.alpha, dtype=torch.float32, device=device)
    model = AIFPINNModel(
        alpha=problem.alpha,
        epsilon=problem.epsilon,
        initial_condition=problem.initial_condition,
        hidden_layers=(128, 128, 64),
        mittag_series_terms=8,
    )
    solver = PINNSolver(
        model,
        operator,
        problem,
        dtype=torch.float32,
        device=device,
    )
    return solver


def evaluate_model(model: torch.nn.Module, t_eval: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t_tensor = torch.as_tensor(t_eval, dtype=torch.float32, device=device).reshape(-1, 1)
        prediction = model(t_tensor)
    return prediction.detach().cpu().numpy().reshape(-1)


def compute_exact(problem: LinearRelaxationProblem, t_eval: np.ndarray, device: torch.device) -> np.ndarray:
    t_tensor = torch.as_tensor(t_eval, dtype=torch.float32, device=device).reshape(-1, 1)
    with torch.no_grad():
        reference = problem.exact_solution(t_tensor)
    if reference is None:
        raise RuntimeError("LinearRelaxationProblem must expose an exact solution.")
    return reference.detach().cpu().numpy().reshape(-1)


def compute_error_metrics(prediction: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    diff = prediction - reference
    denom = np.linalg.norm(reference)
    l2_rel = np.linalg.norm(diff) / denom if denom > 0 else np.inf
    max_err = np.max(np.abs(diff))
    return float(l2_rel), float(max_err)


def render_metrics_table(rows: Iterable[dict[str, str]]) -> str:
    header = f"{'Method':<24} | {'Points':>6} | {'L2 Rel':>10} | {'Max Error':>10} | {'Runtime [s]':>11}"
    separator = "-" * len(header)
    formatted = [header, separator]
    for row in rows:
        formatted.append(
            f"{row['method']:<24} | {row['points']:>6} | {row['l2_rel']:>10} | {row['max_err']:>10} | {row['runtime']:>11}"
        )
    return "\n".join(formatted)


def plot_comparison(
    dense_t: np.ndarray,
    dense_reference: np.ndarray,
    fdm_grid: np.ndarray,
    fdm_solution: np.ndarray,
    pinn_solution: np.ndarray,
    *,
    epsilon: float,
    horizon: float,
    pinn_points: int,
    save_path: Path,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        dense_t,
        dense_reference,
        label="Exact solution (Mittag-Leffler)",
        linestyle="--",
        linewidth=2.0,
        color="#2ca02c",
    )
    ax.plot(
        fdm_grid,
        fdm_solution,
        label=f"FDM (Uniform N={fdm_grid.size})",
        linewidth=1.2,
        color="#d62728",
        marker="o",
        markersize=3.5,
        alpha=0.85,
    )
    ax.plot(
        dense_t,
        pinn_solution,
        label=f"AIF-PINN (Collocation N={pinn_points})",
        linewidth=2.2,
        color="#1f77b4",
    )
    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    ax.set_title("AIF-PINN vs uniform-grid FDM on a stiff fractional relaxation problem")
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.set_xlim(0.0, horizon)

    inset_width = min(5.0 * epsilon, horizon)
    inset = ax.inset_axes([0.5, 0.45, 0.45, 0.45])
    inset.plot(dense_t, dense_reference, linestyle="--", linewidth=1.5, color="#2ca02c")
    inset.plot(fdm_grid, fdm_solution, linewidth=1.0, color="#d62728", marker="o", markersize=3.0)
    inset.plot(dense_t, pinn_solution, linewidth=1.4, color="#1f77b4")
    inset.set_xlim(0.0, inset_width)
    inset.set_ylim(
        min(np.min(fdm_solution), np.min(pinn_solution), np.min(dense_reference)) - 0.05,
        max(np.max(fdm_solution), np.max(pinn_solution), np.max(dense_reference)) + 0.05,
    )
    inset.set_title("Boundary layer zoom", fontsize=9)
    inset.grid(True, linestyle=":", linewidth=0.6)
    ax.indicate_inset_zoom(inset, edgecolor="gray")
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    problem = LinearRelaxationProblem(
        alpha=args.alpha,
        epsilon=args.epsilon,
        initial_condition=args.initial_condition,
        horizon=args.horizon,
    )

    print("Solving stiff case:")
    print(f"  alpha={problem.alpha:.2f}, epsilon={problem.epsilon:.4f}, u0={problem.initial_condition:.2f}")

    # FDM baseline on a uniform grid
    fdm_grid = np.linspace(0.0, problem.horizon, num=args.fdm_points, dtype=np.float64)
    fdm_operator = FractionalDerivativeOperator(fdm_grid, problem.alpha)
    fdm_solver = FDMSolver(problem, fdm_grid, operator=fdm_operator)
    fdm_start = time.perf_counter()
    fdm_solution = fdm_solver.solve()
    fdm_runtime = time.perf_counter() - fdm_start
    fdm_reference = compute_exact(problem, fdm_grid, device)
    fdm_l2, fdm_max = compute_error_metrics(fdm_solution, fdm_reference)

    # AIF-PINN with the same number of collocation points
    pinn_solver = build_pinn_solver(problem, args.num_collocation, device)
    pinn_start = time.perf_counter()
    pinn_solver.train(
        adam_steps=args.adam_steps,
        adam_lr=args.adam_lr,
        adam_weight_decay=args.adam_weight_decay,
        lbfgs_max_iter=args.lbfgs_iter,
        verbose=True,
    )
    pinn_runtime = time.perf_counter() - pinn_start

    dense_t = np.linspace(0.0, problem.horizon, num=args.eval_points, dtype=np.float32)
    dense_reference = compute_exact(problem, dense_t, device)
    pinn_solution = evaluate_model(pinn_solver.model, dense_t, device)
    pinn_l2, pinn_max = compute_error_metrics(pinn_solution, dense_reference)

    metrics_rows = [
        {
            "method": f"FDM (uniform)",
            "points": f"{args.fdm_points:d}",
            "l2_rel": f"{fdm_l2:.3e}",
            "max_err": f"{fdm_max:.3e}",
            "runtime": f"{fdm_runtime:>8.3f}",
        },
        {
            "method": "AIF-PINN",
            "points": f"{args.num_collocation:d}",
            "l2_rel": f"{pinn_l2:.3e}",
            "max_err": f"{pinn_max:.3e}",
            "runtime": f"{pinn_runtime:>8.3f}",
        },
    ]

    print("\nError and runtime comparison:")
    print(render_metrics_table(metrics_rows))

    plot_comparison(
        dense_t,
        dense_reference,
        fdm_grid,
        fdm_solution,
        pinn_solution,
        epsilon=problem.epsilon,
        horizon=problem.horizon,
        pinn_points=args.num_collocation,
        save_path=args.figure,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
