"""Toy linear-softmax model, basin scan, and local stability analysis.

Phase 1 implements the model and numerical scan.  Phase 2 adds the analytic
gradients, reduced bias flow, and symmetric-point Hessian checks.
"""

from theory.basin import make_bias_grid, run_basin_scan, summarize_basin
from theory.stability import (
    StabilitySpec,
    discrete_step_bound,
    entropy_logit_gradient,
    information_logit_gradient,
    information_objective,
    marginal_mode_curvature,
    parameter_gradients,
    parameter_hessian,
    reduced_bias_gradient,
    reduced_bias_hessian,
    simulate_reduced_bias,
    symmetric_b_block_hessian,
)
from theory.toy_model import (
    MixtureData,
    ToySetup,
    TrajectoryResult,
    evaluate_logits,
    initial_parameters,
    make_gaussian_mixture,
    simulate_entropy_minimisation,
    source_metrics,
)

__all__ = [
    "MixtureData",
    "StabilitySpec",
    "ToySetup",
    "TrajectoryResult",
    "discrete_step_bound",
    "entropy_logit_gradient",
    "evaluate_logits",
    "information_logit_gradient",
    "information_objective",
    "initial_parameters",
    "make_bias_grid",
    "make_gaussian_mixture",
    "marginal_mode_curvature",
    "parameter_gradients",
    "parameter_hessian",
    "reduced_bias_gradient",
    "reduced_bias_hessian",
    "run_basin_scan",
    "simulate_entropy_minimisation",
    "simulate_reduced_bias",
    "source_metrics",
    "summarize_basin",
    "symmetric_b_block_hessian",
]
