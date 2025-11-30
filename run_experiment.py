"""Run configurable AIF-PINN experiments for benchmark SPFDE problems."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from aif_pinn import (
    AbstractSPFDE,
    AIFPINNModel,
    DataGenerator,
    FractionalDerivativeOperator,
    LinearRelaxationProblem,
    NonlinearLogisticProblem,
    PINNSolver,
    VariableCoeffProblem,
)


class Visualizer:
    """Utility to compare the AIF-PINN solution with optional reference curves."""

    def __init__(self, *, boundary_layer_width: float, problem_label: str) -> None:
        self.boundary_layer_width = float(boundary_layer_width)
        self.problem_label = problem_label

    def plot(
        self,
        pinn_t: np.ndarray,
        pinn_u: np.ndarray,
        *,
        reference_u: Optional[np.ndarray] = None,
        coefficient_curve: Optional[np.ndarray] = None,
        coefficient_label: str = "Coefficient",
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(pinn_t, pinn_u, label="AIF-PINN", linewidth=2.0, color="#1f77b4")

        if reference_u is not None:
            ax.plot(
                pinn_t,
                reference_u,
                label="Exact / reference",
                linestyle="--",
                color="#d62728",
                linewidth=1.2,
            )

        if self.boundary_layer_width > 0.0:
            extent = min(self.boundary_layer_width, float(pinn_t[-1]))
            ax.axvspan(
                0.0,
                extent,
                color="#ffbb78",
                alpha=0.25,
                label="Boundary layer",
            )

        ax.set_xlabel("t")
        ax.set_ylabel("u(t)")
        ax.set_title(f"AIF-PINN solution: {self.problem_label}")
        ax.grid(True, linestyle=":", linewidth=0.8)

        handles, labels = ax.get_legend_handles_labels()
        if coefficient_curve is not None:
            secondary = ax.twinx()
            coeff_line, = secondary.plot(
                pinn_t,
                coefficient_curve,
                color="#ff7f0e",
                linestyle="--",
                linewidth=1.5,
                label=coefficient_label,
            )
            secondary.set_ylabel(coefficient_label)
            secondary.grid(False)
            handles.append(coeff_line)
            labels.append(coefficient_label)

        ax.legend(handles, labels, loc="best")

        fig.tight_layout()
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300)
            print(f"Figure saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AIF-PINN on benchmark SPFDE problems.")
    parser.add_argument(
        "--problem",
        type=str,
        default="linear",
        choices=("linear", "variable", "logistic"),
        help="Which benchmark problem to solve.",
    )
    parser.add_argument("--num-collocation", type=int, default=512, help="Number of collocation points.")
    parser.add_argument("--adam-steps", type=int, default=3000, help="Number of Adam iterations.")
    parser.add_argument("--lbfgs-iter", type=int, default=2000, help="Number of L-BFGS iterations.")
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Fractional order alpha.")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Relaxation parameter epsilon.")
    parser.add_argument("--horizon", type=float, default=5.0, help="Time horizon for training and evaluation.")
    parser.add_argument(
        "--initial-condition",
        type=float,
        default=None,
        help="Override the default initial condition for the chosen problem.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="torch device to run on.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures") / "aif_pinn_benchmark.png",
        help="Where to save the plot.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display the matplotlib window.")
    return parser.parse_args()


def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_solver(problem: AbstractSPFDE, args: argparse.Namespace, device: torch.device) -> tuple[PINNSolver, np.ndarray]:
    generator = DataGenerator(problem)
    grid = generator.generate_grid(args.num_collocation)
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
    return solver, grid


def evaluate_model(model: torch.nn.Module, t_eval: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t_tensor = torch.as_tensor(t_eval, dtype=torch.float32, device=device).reshape(-1, 1)
        prediction = model(t_tensor)
        return prediction.cpu().numpy().reshape(-1)


def compute_reference(problem: AbstractSPFDE, t_eval: np.ndarray, device: torch.device) -> Optional[np.ndarray]:
    t_tensor = torch.as_tensor(t_eval, dtype=torch.float32, device=device).reshape(-1, 1)
    with torch.no_grad():
        reference = problem.exact_solution(t_tensor)
    if reference is None:
        return None
    return reference.detach().cpu().numpy().reshape(-1)


def create_problem(args: argparse.Namespace) -> AbstractSPFDE:
    registry: dict[str, type[AbstractSPFDE]] = {
        "linear": LinearRelaxationProblem,
        "variable": VariableCoeffProblem,
        "logistic": NonlinearLogisticProblem,
    }
    defaults = {"linear": 1.0, "variable": 0.5, "logistic": 0.1}
    problem_cls = registry[args.problem]
    initial_condition = args.initial_condition if args.initial_condition is not None else defaults[args.problem]
    return problem_cls(
        alpha=args.alpha,
        epsilon=args.epsilon,
        initial_condition=initial_condition,
        horizon=args.horizon,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    problem = create_problem(args)
    solver, grid = build_solver(problem, args, device)
    history = solver.train(
        adam_steps=args.adam_steps,
        adam_lr=args.adam_lr,
        lbfgs_max_iter=args.lbfgs_iter,
        verbose=True,
    )

    print(f"Training complete. Final loss: {history['lbfgs'][-1] if history['lbfgs'] else history['adam'][-1]:.3e}")

    dense_t = np.linspace(0.0, problem.horizon, num=1000, dtype=np.float32)
    pinn_solution = evaluate_model(solver.model, dense_t, device)
    reference_solution = compute_reference(problem, dense_t, device)
    coefficient_curve = None
    if args.problem == "variable":
        coefficient_curve = 1.0 + 0.5 * np.sin(4.0 * np.pi * dense_t)

    visualizer = Visualizer(boundary_layer_width=problem.boundary_layer_width(), problem_label=args.problem.title())
    visualizer.plot(
        dense_t,
        pinn_solution,
        reference_u=reference_solution,
        coefficient_curve=coefficient_curve,
        coefficient_label="1 + 0.5 sin(4*pi*t)",
        save_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
