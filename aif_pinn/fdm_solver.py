"""Finite difference baseline solver for the linear SPFDE."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .operators import FractionalDerivativeOperator
from .problem import AbstractSPFDE


class FDMSolver:
    """Solve epsilon^alpha * D^alpha u + u = 0 with a uniform-grid L1 scheme."""

    def __init__(
        self,
        problem: AbstractSPFDE,
        grid: np.ndarray,
        *,
        operator: Optional[FractionalDerivativeOperator] = None,
    ) -> None:
        flat_grid = np.asarray(grid, dtype=np.float64).reshape(-1)
        if flat_grid.size < 2:
            raise ValueError("grid must contain at least two points.")

        self.problem = problem
        self.grid = flat_grid
        self.operator = operator or FractionalDerivativeOperator(flat_grid, problem.alpha)

    def assemble_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build A and b for (epsilon^alpha P + I) u = b with u(0) = u0."""

        matrix_P = np.array(self.operator.compute_matrix_P(), copy=True)
        n_points = matrix_P.shape[0]
        eps_alpha = float(pow(self.problem.epsilon, self.problem.alpha))

        A = eps_alpha * matrix_P + np.eye(n_points, dtype=np.float64)
        b = np.zeros(n_points, dtype=np.float64)

        # Enforce the initial condition strongly at t = 0.
        A[0, :] = 0.0
        A[0, 0] = 1.0
        b[0] = self.problem.initial_condition

        return A, b

def solve(self) -> np.ndarray:
    """Solve the assembled linear system for the nodal values."""

    A, b = self.assemble_system()
    solution = np.linalg.solve(A, b)
    return solution


def solve_variable_coeff(problem: AbstractSPFDE, grid_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Solve epsilon^alpha D^alpha u + a(t) u = f(t) on a uniform grid."""

    if grid_points < 2:
        raise ValueError("grid_points must be at least 2.")

    grid = np.linspace(0.0, problem.horizon, num=grid_points, dtype=np.float64)
    operator = FractionalDerivativeOperator(grid, problem.alpha)
    matrix_P = np.array(operator.compute_matrix_P(), copy=True)
    n_points = matrix_P.shape[0]
    eps_alpha = float(pow(problem.epsilon, problem.alpha))

    coeff_fn = getattr(problem, "reaction_coefficient", None)
    forcing_fn = getattr(problem, "forcing", None)
    if coeff_fn is None or forcing_fn is None:
        raise AttributeError("problem must define reaction_coefficient and forcing methods.")

    coeff_values = np.asarray(coeff_fn(grid), dtype=np.float64).reshape(-1)
    forcing_values = np.asarray(forcing_fn(grid), dtype=np.float64).reshape(-1)
    if coeff_values.shape[0] != n_points or forcing_values.shape[0] != n_points:
        raise ValueError("Coefficient and forcing evaluations must match the grid size.")

    A = eps_alpha * matrix_P
    diag_idx = np.arange(n_points)
    A[diag_idx, diag_idx] += coeff_values
    b = forcing_values.astype(np.float64, copy=True)

    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = problem.initial_condition

    solution = np.linalg.solve(A, b)
    return grid, solution


__all__ = ["FDMSolver", "solve_variable_coeff"]
