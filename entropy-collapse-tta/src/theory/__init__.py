"""Theory module export."""

from src.theory.data import generate_gaussian_mixture
from src.theory.model import LinearSoftmaxModel
from src.theory.dynamics import simulate_dynamics, is_collapsed_state
from src.theory.basin import (
    find_critical_delta_bisection,
    sweep_bias_basin,
    compute_phase_diagram_2d,
)

__all__ = [
    "generate_gaussian_mixture",
    "LinearSoftmaxModel",
    "simulate_dynamics",
    "is_collapsed_state",
    "find_critical_delta_bisection",
    "sweep_bias_basin",
    "compute_phase_diagram_2d",
]
