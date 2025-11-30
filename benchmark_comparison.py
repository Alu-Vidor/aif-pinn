"""Benchmark the robustness of AIF-PINN across multiple epsilon values."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

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


@dataclass
class BenchmarkResult:
    """Container for per-epsilon benchmark measurements."""

    epsilon: float
    alpha: float
    l2_error: float
    max_error: float
    adam_steps: int
    lbfgs_iter: int
    t: np.ndarray
    pinn_solution: np.ndarray
    reference_solution: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AIF-PINN accuracy for several epsilon values.")
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.1, 0.05, 0.01],
        help="List of epsilon values to benchmark.",
    )
    parser.add_argument("--alpha", type=float, default=0.8, help="Fractional order alpha.")
    parser.add_argument("--num-collocation", type=int, default=512, help="Number of collocation points.")
    parser.add_argument("--adam-steps", type=int, default=3000, help="Number of Adam iterations.")
    parser.add_argument("--lbfgs-iter", type=int, default=2000, help="Number of L-BFGS iterations.")
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--adam-weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument("--horizon", type=float, default=5.0, help="Time horizon for training and evaluation.")
    parser.add_argument("--device", type=str, default="cpu", help="torch device to run on.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--eval-points", type=int, default=1000, help="Number of evaluation points for error metrics.")
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures") / "benchmark_comparison.png",
        help="Where to save the combined comparison plot.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display the matplotlib window.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_solver(config: ProblemConfig, args: argparse.Namespace, device: torch.device) -> PINNSolver:
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
    return solver


def evaluate_model(model: torch.nn.Module, t_eval: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t_tensor = torch.as_tensor(t_eval, dtype=torch.float32, device=device).reshape(-1, 1)
        prediction = model(t_tensor)
        return prediction.cpu().numpy().reshape(-1)


def mittag_leffler_reference(t_eval: np.ndarray, config: ProblemConfig) -> np.ndarray:
    alpha = config.alpha
    eps = torch.tensor(config.epsilon, dtype=torch.float32)
    u0 = torch.tensor(config.initial_condition, dtype=torch.float32)
    t_tensor = torch.as_tensor(t_eval, dtype=torch.float32).reshape(-1, 1)
    eps_alpha = torch.pow(eps, alpha)
    z = -torch.pow(torch.clamp(t_tensor, min=0.0), alpha) / eps_alpha
    ml = mittag_leffler_approx(z, alpha, series_terms=12, switch_threshold=1.0)
    return (u0 * ml.squeeze(-1)).cpu().numpy()


def compute_error_metrics(prediction: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    diff = prediction - reference
    denom = np.linalg.norm(reference)
    l2_rel = np.linalg.norm(diff) / denom if denom > 0 else np.inf
    max_err = np.max(np.abs(diff))
    return float(l2_rel), float(max_err)


def run_single_benchmark(
    epsilon: float,
    args: argparse.Namespace,
    device: torch.device,
) -> BenchmarkResult:
    config = ProblemConfig(
        alpha=args.alpha,
        epsilon=epsilon,
        lambda_coeff=1.0,
        initial_condition=-0.5,
        horizon=args.horizon,
    )
    solver = build_solver(config, args, device)
    history = solver.train(
        adam_steps=args.adam_steps,
        adam_lr=args.adam_lr,
        adam_weight_decay=args.adam_weight_decay,
        lbfgs_max_iter=args.lbfgs_iter,
        verbose=True,
    )
    dense_t = np.linspace(0.0, config.horizon, num=args.eval_points, dtype=np.float32)
    pinn_solution = evaluate_model(solver.model, dense_t, device)
    reference_solution = mittag_leffler_reference(dense_t, config)
    l2_error, max_error = compute_error_metrics(pinn_solution, reference_solution)

    adam_steps_ran = len(history["adam"]) if history["adam"] else 0
    lbfgs_iters_ran = args.lbfgs_iter if history["lbfgs"] else 0

    return BenchmarkResult(
        epsilon=epsilon,
        alpha=config.alpha,
        l2_error=l2_error,
        max_error=max_error,
        adam_steps=adam_steps_ran,
        lbfgs_iter=lbfgs_iters_ran,
        t=dense_t,
        pinn_solution=pinn_solution,
        reference_solution=reference_solution,
    )


def render_table(results: Iterable[BenchmarkResult]) -> str:
    header = f"{'Epsilon':>8} | {'Alpha':>5} | {'L2 Error':>10} | {'Max Error':>10} | {'Adam Steps':>10} | {'L-BFGS Iter':>11}"
    separator = "-" * len(header)
    rows = []
    for res in results:
        rows.append(
            f"{res.epsilon:8.3f} | {res.alpha:5.2f} | {res.l2_error:10.3e} | {res.max_error:10.3e} | "
            f"{res.adam_steps:10d} | {res.lbfgs_iter:11d}"
        )
    return "\n".join([header, separator, *rows])


def plot_results(results: list[BenchmarkResult], args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("viridis", len(results))

    for idx, res in enumerate(results):
        color = cmap(idx)
        label_eps = f"epsilon={res.epsilon:.3f}"
        ax.plot(res.t, res.pinn_solution, color=color, linewidth=2.0, label=f"AIF-PINN {label_eps}")
        ax.plot(
            res.t,
            res.reference_solution,
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=f"Reference {label_eps}",
        )

    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    ax.set_title("AIF-PINN vs Mittag-Leffler reference for varying epsilon")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()

    if args.figure is not None:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=300)
        print(f"Figure saved to {args.figure}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    results: list[BenchmarkResult] = []
    for eps in args.epsilons:
        print("\n" + "=" * 72)
        print(f"Training AIF-PINN for epsilon={eps:.3f}")
        result = run_single_benchmark(eps, args, device)
        results.append(result)
        print(
            f"Completed epsilon={eps:.3f} | L2 Error: {result.l2_error:.3e}, "
            f"Max Error: {result.max_error:.3e}"
        )

    if not results:
        print("No epsilon values were provided. Nothing to benchmark.")
        return

    print("\nBenchmark summary:")
    print(render_table(results))
    plot_results(results, args)


if __name__ == "__main__":
    main()
