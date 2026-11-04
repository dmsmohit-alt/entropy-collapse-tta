"""Numerical basin of attraction mapping and critical threshold estimation."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from src.theory.data import generate_gaussian_mixture
from src.theory.model import LinearSoftmaxModel
from src.theory.dynamics import simulate_dynamics, simulate_bias_dynamics_fast, check_collapse_criteria


def find_critical_delta_bisection(
    model_template: LinearSoftmaxModel,
    X: torch.Tensor,
    y: torch.Tensor,
    delta_low: float = 0.0,
    delta_high: float = 4.0,
    tol: float = 1e-3,
    max_iter: int = 15,
    num_steps: int = 600,
    lr: float = 0.05,
    adapt_mode: str = "bias_only",
    collapse_definition: str = "standard",
    fast: bool = True,
) -> float:
    """Find the critical initial bias offset delta* separating stability from collapse via bisection.

    Args:
        model_template: Fitted base model (delta=0).
        X: Synthetic inputs.
        y: Labels for evaluation.
        delta_low: Known non-collapsing delta (e.g. 0.0).
        delta_high: Known collapsing delta (e.g. 4.0).
        tol: Precision tolerance for bisection.
        max_iter: Maximum bisection steps.
        num_steps: Steps to run gradient descent for each trial.
        lr: Learning rate.
        adapt_mode: "bias_only" or "weights_and_bias".
        collapse_definition: Which collapse rule to apply ('standard', 'def_a', 'def_b', 'def_c').
        fast: If True and bias_only, uses optimized tensor simulation.

    Returns:
        Estimated critical delta*.
    """
    low = delta_low
    high = delta_high

    # Precompute F = X @ W.T once for speed if fast bias_only
    use_fast = fast and (adapt_mode == "bias_only")
    if use_fast:
        with torch.no_grad():
            F = X @ model_template.weight.T

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        if use_fast:
            b_init = model_template.bias.clone()
            b_init[0] += mid
            res = simulate_bias_dynamics_fast(
                precomputed_F=F,
                initial_bias=b_init,
                y_eval=y,
                learning_rate=lr,
                num_steps=num_steps,
                collapse_definition=collapse_definition,
                log_every=max(1, num_steps // 20),
                early_stop_on_collapse=True,
            )
        else:
            m = model_template.clone()
            m.apply_bias_offset(delta=mid, class_idx=0)
            res = simulate_dynamics(
                m,
                X,
                y,
                learning_rate=lr,
                num_steps=num_steps,
                adapt_mode=adapt_mode,
                objective="tent",
                collapse_definition=collapse_definition,
            )

        if res["is_collapsed"]:
            high = mid
        else:
            low = mid

        if (high - low) < tol:
            break

    return (low + high) / 2.0


def sweep_bias_basin(
    model_template: LinearSoftmaxModel,
    X: torch.Tensor,
    y: torch.Tensor,
    delta_grid: np.ndarray,
    learning_rate: float = 0.05,
    num_steps: int = 600,
    adapt_mode: str = "bias_only",
    objective: str = "tent",
    lambda_im: float = 1.0,
    collapse_definition: str = "standard",
    fast: bool = True,
) -> Dict[str, Any]:
    """Sweep initial bias delta over a 1D grid and record outcome metrics.

    Args:
        model_template: Base model (W, b).
        X: Dataset features.
        y: Ground-truth labels.
        delta_grid: Array of delta values to test.
        learning_rate: Step size eta.
        num_steps: Simulation steps.
        adapt_mode: "bias_only" or "weights_and_bias".
        objective: "tent" or "tent_im".
        lambda_im: Regularisation strength for tent_im.
        collapse_definition: Rule name for collapse determination.
        fast: Use vectorized acceleration for bias-only tent.

    Returns:
        Dict with arrays for delta, collapsed (bool), final_h_pbar, final_top_fraction, final_acc, final_grad_norm.
    """
    collapsed_list = []
    final_h_pbar_list = []
    final_top_frac_list = []
    final_acc_list = []
    final_grad_norm_list = []

    use_fast = fast and (adapt_mode == "bias_only") and (objective == "tent")
    if use_fast:
        with torch.no_grad():
            F = X @ model_template.weight.T

    for delta in delta_grid:
        if use_fast:
            b_init = model_template.bias.clone()
            b_init[0] += float(delta)
            res = simulate_bias_dynamics_fast(
                precomputed_F=F,
                initial_bias=b_init,
                y_eval=y,
                learning_rate=learning_rate,
                num_steps=num_steps,
                collapse_definition=collapse_definition,
                log_every=max(1, num_steps // 20),
                early_stop_on_collapse=True,
            )
        else:
            m = model_template.clone()
            m.apply_bias_offset(delta=float(delta), class_idx=0)
            res = simulate_dynamics(
                m,
                X,
                y,
                learning_rate=learning_rate,
                num_steps=num_steps,
                adapt_mode=adapt_mode,
                objective=objective,
                lambda_im=lambda_im,
                collapse_definition=collapse_definition,
            )

        collapsed_list.append(res["is_collapsed"])
        final_h_pbar_list.append(res["final_marginal_entropy"])
        final_top_frac_list.append(res["final_top_fraction"])
        final_acc_list.append(res["final_accuracy"])
        final_grad_norm_list.append(res["final_grad_norm"])

    return {
        "deltas": np.array(delta_grid),
        "collapsed": np.array(collapsed_list, dtype=bool),
        "final_h_pbar": np.array(final_h_pbar_list),
        "final_top_fraction": np.array(final_top_frac_list),
        "final_accuracy": np.array(final_acc_list),
        "final_grad_norm": np.array(final_grad_norm_list),
    }


def compute_phase_diagram_2d(
    separations: np.ndarray,
    deltas: np.ndarray,
    num_classes: int = 3,
    dim: int = 2,
    samples_per_class: int = 400,
    noise_std: float = 1.0,
    learning_rate: float = 0.05,
    num_steps: int = 500,
    seed: int = 42,
    collapse_definition: str = "standard",
) -> Dict[str, Any]:
    """Compute 2D basin boundary phase diagram: separation vs initial bias delta."""
    phase_grid = np.zeros((len(separations), len(deltas)), dtype=bool)

    for i, sep in enumerate(separations):
        X, y, centroids = generate_gaussian_mixture(
            num_classes=num_classes,
            dim=dim,
            samples_per_class=samples_per_class,
            separation=float(sep),
            noise_std=noise_std,
            seed=seed,
        )
        base_model = LinearSoftmaxModel(num_classes=num_classes, dim=dim)
        base_model.fit_centroid_solution(centroids, scale=float(sep))

        with torch.no_grad():
            F = X @ base_model.weight.T

        for j, delta in enumerate(deltas):
            b_init = base_model.bias.clone()
            b_init[0] += float(delta)
            res = simulate_bias_dynamics_fast(
                precomputed_F=F,
                initial_bias=b_init,
                y_eval=y,
                learning_rate=learning_rate,
                num_steps=num_steps,
                collapse_definition=collapse_definition,
                log_every=max(1, num_steps // 10),
            )
            phase_grid[i, j] = res["is_collapsed"]

    return {
        "separations": separations,
        "deltas": deltas,
        "phase_grid": phase_grid,
    }
