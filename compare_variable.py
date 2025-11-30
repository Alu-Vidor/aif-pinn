"""Compare AIF-PINN and FDM on the VariableCoeffProblem with segmented errors."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from aif_pinn import (
    AIFPINNModel,
    DataGenerator,
    FractionalDerivativeOperator,
    PINNSolver,
    VariableCoeffProblem,
    solve_variable_coeff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segmented comparison of AIF-PINN and FDM on VariableCoeffProblem.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Fractional order alpha.")
    parser.add_argument("--epsilon", type=float, default=0.02, help="Singular perturbation parameter epsilon.")
    parser.add_argument("--horizon", type=float, default=5.0, help="Time horizon.")
    parser.add_argument("--initial-condition", type=float, default=0.5, help="Initial value u(0).")
    parser.add_argument("--fdm-points", type=int, default=200, help="Number of uniform grid points for FDM.")
    parser.add_argument("--pinn-collocation", type=int, default=200, help="Collocation points for the compared AIF-PINN.")
    parser.add_argument("--reference-collocation", type=int, default=2000, help="Collocation points for the quasi-exact PINN.")
    parser.add_argument("--adam-steps", type=int, default=2500, help="Adam iterations for the compared AIF-PINN.")
    parser.add_argument("--lbfgs-iter", type=int, default=1500, help="L-BFGS iterations for the compared AIF-PINN.")
    parser.add_argument("--reference-adam-steps", type=int, default=4000, help="Adam iterations for the quasi-exact PINN.")
    parser.add_argument("--reference-lbfgs-iter", type=int, default=2500, help="L-BFGS iterations for the quasi-exact PINN.")
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--adam-weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument("--eval-points", type=int, default=1500, help="Dense evaluation points for plotting and metrics.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to run the models on.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures") / "variable_coeff_comparison.png",
        help="Where to save the resulting plot.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display the matplotlib window.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_solver(problem: VariableCoeffProblem, num_points: int, device: torch.device) -> PINNSolver:
    generator = DataGenerator(problem)
    grid = generator.generate_grid(num_points)
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


def train_solver(
    solver: PINNSolver,
    *,
    adam_steps: int,
    adam_lr: float,
    adam_weight_decay: float,
    lbfgs_iter: int,
) -> None:
    solver.train(
        adam_steps=adam_steps,
        adam_lr=adam_lr,
        adam_weight_decay=adam_weight_decay,
        lbfgs_max_iter=lbfgs_iter,
        verbose=True,
    )


def evaluate_model(model: torch.nn.Module, t_eval: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t_tensor = torch.as_tensor(t_eval, dtype=torch.float32, device=device).reshape(-1, 1)
        prediction = model(t_tensor)
    return prediction.detach().cpu().numpy().reshape(-1)


def compute_segmented_metrics(
    t: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    boundary_limit: float,
) -> dict[str, float]:
    abs_diff = np.abs(prediction - reference)
    boundary_mask = t <= boundary_limit
    outer_mask = t > boundary_limit

    bl_max = float(np.max(abs_diff[boundary_mask])) if np.any(boundary_mask) else 0.0
    outer_max = float(np.max(abs_diff[outer_mask])) if np.any(outer_mask) else 0.0
    rmse = float(np.sqrt(np.mean(np.square(prediction - reference))))

    return {"bl_max": bl_max, "outer_max": outer_max, "l2": rmse}


def render_metrics_table(rows: list[dict[str, float]]) -> str:
    header = "Method    | BL-Max Error | Outer-Max Error | Total L2 Error"
    separator = "-" * len(header)
    formatted = [header, separator]
    for row in rows:
        formatted.append(
            f"{row['method']:<10} | {row['bl_max']:>12.3e} | {row['outer_max']:>15.3e} | {row['l2']:>15.3e}"
        )
    return "\n".join(formatted)


def plot_results(
    t_dense: np.ndarray,
    reference: np.ndarray,
    pinn_solution: np.ndarray,
    fdm_grid: np.ndarray,
    fdm_solution: np.ndarray,
    *,
    coefficient: np.ndarray,
    boundary_limit: float,
    pinn_points: int,
    fdm_points: int,
    figure_path: Path,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        t_dense,
        reference,
        label="Quasi-exact PINN",
        linestyle="--",
        color="#2ca02c",
        linewidth=2.0,
    )
    ax.plot(
        t_dense,
        pinn_solution,
        label=f"AIF-PINN (N={pinn_points})",
        color="#1f77b4",
        linewidth=2.2,
    )
    ax.plot(
        fdm_grid,
        fdm_solution,
        label=f"FDM (Uniform N={fdm_points})",
        color="#d62728",
        linewidth=1.0,
        marker="o",
        markersize=3.0,
        alpha=0.85,
    )

    layer_extent = min(boundary_limit, float(t_dense[-1]))
    if layer_extent > 0.0:
        ax.axvspan(0.0, layer_extent, color="#ffbb78", alpha=0.25, label="Boundary layer")

    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    ax.set_title("Variable coefficients: AIF-PINN vs FDM")
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.set_xlim(0.0, float(t_dense[-1]))

    secondary = ax.twinx()
    secondary.plot(
        t_dense,
        coefficient,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.4,
        label="a(t)",
    )
    secondary.set_ylabel("a(t)")
    secondary.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    coeff_line = secondary.get_lines()[0]
    handles.append(coeff_line)
    labels.append("a(t)")
    ax.legend(handles, labels, loc="best")

    fig.tight_layout()
    if figure_path is not None:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=300)
        print(f"Figure saved to {figure_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    problem = VariableCoeffProblem(
        alpha=args.alpha,
        epsilon=args.epsilon,
        initial_condition=args.initial_condition,
        horizon=args.horizon,
    )
    boundary_limit = 5.0 * problem.epsilon

    print("Training quasi-exact PINN (dense collocation)...")
    reference_solver = build_solver(problem, args.reference_collocation, device)
    train_solver(
        reference_solver,
        adam_steps=args.reference_adam_steps,
        adam_lr=args.adam_lr,
        adam_weight_decay=args.adam_weight_decay,
        lbfgs_iter=args.reference_lbfgs_iter,
    )

    t_dense = np.linspace(0.0, problem.horizon, num=args.eval_points, dtype=np.float32)
    reference_dense = evaluate_model(reference_solver.model, t_dense, device)

    print("\nTraining compared AIF-PINN...")
    pinn_solver = build_solver(problem, args.pinn_collocation, device)
    train_solver(
        pinn_solver,
        adam_steps=args.adam_steps,
        adam_lr=args.adam_lr,
        adam_weight_decay=args.adam_weight_decay,
        lbfgs_iter=args.lbfgs_iter,
    )
    pinn_dense = evaluate_model(pinn_solver.model, t_dense, device)

    print("\nSolving FDM baseline with variable coefficients...")
    fdm_grid, fdm_solution = solve_variable_coeff(problem, args.fdm_points)
    reference_on_fdm = evaluate_model(reference_solver.model, fdm_grid.astype(np.float32), device)

    pinn_metrics = compute_segmented_metrics(
        t_dense,
        pinn_dense,
        reference_dense,
        boundary_limit=boundary_limit,
    )
    fdm_metrics = compute_segmented_metrics(
        fdm_grid,
        fdm_solution,
        reference_on_fdm,
        boundary_limit=boundary_limit,
    )

    table_rows = [
        {"method": "FDM (Unif)", **fdm_metrics},
        {"method": "AIF-PINN", **pinn_metrics},
    ]
    print("\nSegmented error comparison:")
    print(render_metrics_table(table_rows))

    coefficient_curve = problem.reaction_coefficient(t_dense)
    plot_results(
        t_dense,
        reference_dense,
        pinn_dense,
        fdm_grid,
        fdm_solution,
        coefficient=coefficient_curve,
        boundary_limit=boundary_limit,
        pinn_points=args.pinn_collocation,
        fdm_points=args.fdm_points,
        figure_path=args.figure,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
