"""Convenience exports for the core AIF-PINN components."""

from .data import DataGenerator
from .fdm_solver import FDMSolver, solve_variable_coeff
from .mittag import mittag_leffler_approx
from .model import AIFPINNModel
from .operators import FractionalDerivativeOperator
from .problem import AbstractSPFDE
from .problems_zoo import (
    LinearRelaxationProblem,
    NonlinearLogisticProblem,
    VariableCoeffProblem,
)
from .solver import PINNSolver

__all__ = [
    "AbstractSPFDE",
    "DataGenerator",
    "FractionalDerivativeOperator",
    "AIFPINNModel",
    "PINNSolver",
    "mittag_leffler_approx",
    "LinearRelaxationProblem",
    "VariableCoeffProblem",
    "NonlinearLogisticProblem",
    "FDMSolver",
    "solve_variable_coeff",
]
