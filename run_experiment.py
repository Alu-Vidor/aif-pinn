"""Run the Solow growth with memory experiment using the AIF-PINN solver."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from aif_pinn import (
    AIFPINNModel,
    DataGenerator,
    FractionalDerivativeOperator,
    PINNSolver,
    ProblemConfig,
    mittag_leffler_approx,
)


class Visualizer:
    """Utility to compare the AIF-PINN solution with a reference curve."""

    def __init__(self, boundary_layer_extent: float) -> None:
        self.boundary_layer_extent = float(boundary_layer_extent)

    def plot(
        self,
        pinn_t: np.ndarray,
        pinn_u: np.ndarray,
        *,
        reference_t: Optional[np.ndarray] = None,
        reference_u: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(pinn_t, pinn_u, label="AIF-PINN", linewidth=2.0, color="#1f77b4")

        if reference_t is not None and reference_u is not None:
            ax.plot(
                reference_t,
                reference_u,
                label="Mittag-Leffler reference",
                linestyle="--",
                color="#d62728",
                linewidth=1.5,
            )

        # Highlight the boundary layer to show rapid relaxation.
        ax.axvspan(
            0.0,
            self.boundary_layer_extent,
            color="#ffbb78",
            alpha=0.3,
            label="Boundary layer",
        )

        ax.set_xlabel("t")
        ax.set_ylabel("u(t)")
        ax.set_title("Solow growth with memory: AIF-PINN vs reference")
        ax.grid(True, which="both", linestyle=":")
        ax.legend(loc="best")

        # A symlog scale emphasises the heavy tail without hiding early recovery.
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.annotate(
            "Fast recovery\n(pogranichny sloy)",
            xy=(self.boundary_layer_extent * 0.6, pinn_u[1]),
            xytext=(self.boundary_layer_extent * 1.2, pinn_u[1] * 0.5),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10,
        )
        ax.annotate(
            "Power-law tail\n(fractional memory)",
            xy=(pinn_t[-1], pinn_u[-1]),
            xytext=(pinn_t[-1] * 0.6, pinn_u[-1] * 0.8),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10,
        )

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
    parser = argparse.ArgumentParser(description="Train AIF-PINN on the fractional Solow growth model.")
    parser.add_argument("--num-collocation", type=int, default=512, help="Number of collocation points.")
    parser.add_argument("--adam-steps", type=int, default=3000, help="Number of Adam iterations.")
    parser.add_argument("--lbfgs-iter", type=int, default=2000, help="Number of L-BFGS iterations.")
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Relaxation parameter epsilon.")
    parser.add_argument("--horizon", type=float, default=5.0, help="Time horizon for training and evaluation.")
    parser.add_argument("--device", type=str, default="cpu", help="torch device to run on.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures") / "solow_aif_pinn.png",
        help="Where to save the plot.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display the matplotlib window.")
    return parser.parse_args()


def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_solver(config: ProblemConfig, args: argparse.Namespace, device: torch.device) -> tuple[PINNSolver, np.ndarray]:
    generator = DataGenerator(config)
    grid = generator.generate_grid(args.num_collocation)
    operator = FractionalDerivativeOperator(grid, config.alpha, dtype=torch.float32, device=device)
    model = AIFPINNModel(
        alpha=config.alpha,
        epsilon=config.epsilon,
        initial_condition=config.initial_condition,
        hidden_layers=(128, 128, 64),
        mittag_series_terms=8,
    )
    solver = PINNSolver(
        model,
        operator,
        config,
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


def mittag_leffler_reference(t_eval: np.ndarray, config: ProblemConfig) -> Optional[np.ndarray]:
    alpha = config.alpha
    eps = torch.tensor(config.epsilon, dtype=torch.float32)
    u0 = torch.tensor(config.initial_condition, dtype=torch.float32)
    t_tensor = torch.as_tensor(t_eval, dtype=torch.float32).reshape(-1, 1)
    eps_alpha = torch.pow(eps, alpha)
    z = -torch.pow(torch.clamp(t_tensor, min=0.0), alpha) / eps_alpha
    ml = mittag_leffler_approx(z, alpha, series_terms=12, switch_threshold=1.0)
    return (u0 * ml.squeeze(-1)).cpu().numpy()


def main() -> None:
    args = parse_args()
    seed_everything()

    config = ProblemConfig(
        alpha=0.8,
        epsilon=args.epsilon,
        lambda_coeff=1.0,
        initial_condition=-0.5,
        horizon=args.horizon,
    )

    device = torch.device(args.device)
    solver, grid = build_solver(config, args, device)
    history = solver.train(
        adam_steps=args.adam_steps,
        adam_lr=args.adam_lr,
        lbfgs_max_iter=args.lbfgs_iter,
        verbose=True,
    )

    print(f"Training complete. Final loss: {history['lbfgs'][-1] if history['lbfgs'] else history['adam'][-1]:.3e}")

    dense_t = np.linspace(0.0, config.horizon, num=1000, dtype=np.float32)
    pinn_solution = evaluate_model(solver.model, dense_t, device)
    reference_solution = mittag_leffler_reference(dense_t, config)

    visualizer = Visualizer(boundary_layer_extent=config.boundary_layer_extent)
    visualizer.plot(
        dense_t,
        pinn_solution,
        reference_t=dense_t if reference_solution is not None else None,
        reference_u=reference_solution,
        save_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
