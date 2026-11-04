"""Unit tests for theoretical toy dynamics, gradient checks, and collapse properties."""

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.theory.data import generate_gaussian_mixture
from src.theory.model import LinearSoftmaxModel
from src.theory.dynamics import simulate_dynamics, is_collapsed_state
from src.metrics.entropy import compute_conditional_entropy, compute_marginal_entropy


class TestTheoryDynamics:
    """Test suite for gradient dynamics, stability, and finite differences."""

    def test_bias_gradient_finite_differences_and_autograd(self) -> None:
        """Verify analytical bias gradient E_x[-p_k(ln p_k + H)] matches autograd & finite differences."""
        torch.manual_seed(42)
        X, _, centroids = generate_gaussian_mixture(
            num_classes=3, dim=2, samples_per_class=100, separation=2.0, noise_std=1.0, seed=42
        )
        model = LinearSoftmaxModel(num_classes=3, dim=2)
        model.fit_centroid_solution(centroids, scale=2.0)
        model.apply_bias_offset(delta=0.5, class_idx=0)

        # 1. Autograd gradient
        model.bias.requires_grad = True
        logits = model(X)
        loss = compute_conditional_entropy(logits, reduction="mean")
        loss.backward()
        autograd_grad = model.bias.grad.clone()

        # 2. Analytical formula: dL/db_k = E_x[ -p_k (ln p_k + H(p)) ]
        with torch.no_grad():
            p = F.softmax(logits, dim=-1)
            ln_p = F.log_softmax(logits, dim=-1)
            H = -(p * ln_p).sum(dim=-1, keepdim=True)
            analytical_per_sample = -p * (ln_p + H)
            analytical_grad = analytical_per_sample.mean(dim=0)

        assert torch.allclose(autograd_grad, analytical_grad, atol=1e-5), (
            f"Autograd {autograd_grad} != Analytical {analytical_grad}"
        )

        # 3. Numerical finite differences: (L(b + eps) - L(b - eps)) / (2 * eps)
        eps = 1e-4
        numerical_grad = torch.zeros_like(model.bias)
        with torch.no_grad():
            for k in range(model.num_classes):
                b_orig = model.bias[k].item()

                model.bias[k] = b_orig + eps
                l_plus = compute_conditional_entropy(model(X), reduction="mean").item()

                model.bias[k] = b_orig - eps
                l_minus = compute_conditional_entropy(model(X), reduction="mean").item()

                model.bias[k] = b_orig  # restore
                numerical_grad[k] = (l_plus - l_minus) / (2.0 * eps)

        assert torch.allclose(autograd_grad, numerical_grad, atol=1e-4), (
            f"Autograd {autograd_grad} != Finite Difference {numerical_grad}"
        )

    def test_delta_zero_symmetric_stays_near_uniform(self) -> None:
        """Verify delta = 0 on balanced symmetric data remains stable (H(p_bar) stays near ln(K))."""
        X, y, centroids = generate_gaussian_mixture(
            num_classes=3, dim=2, samples_per_class=300, separation=2.0, noise_std=1.0, seed=42
        )
        model = LinearSoftmaxModel(num_classes=3, dim=2)
        model.fit_centroid_solution(centroids, scale=2.0)

        res = simulate_dynamics(
            model, X, y, learning_rate=0.02, num_steps=150, adapt_mode="bias_only", objective="tent"
        )

        max_ent = math.log(3)
        # Should NOT collapse
        assert not res["is_collapsed"]
        # Marginal entropy should remain close to max entropy (within 10%)
        assert res["final_marginal_entropy"] > 0.85 * max_ent
        # Predictions balanced: top fraction < 0.50 for 3 classes
        assert res["final_top_fraction"] < 0.50

    def test_entropy_monotone_decrease_small_step(self) -> None:
        """Verify that under small step GD (eta=0.005), loss E[H(p)] decreases monotonically."""
        X, y, centroids = generate_gaussian_mixture(
            num_classes=3, dim=2, samples_per_class=200, separation=2.0, noise_std=1.0, seed=42
        )
        model = LinearSoftmaxModel(num_classes=3, dim=2)
        model.fit_centroid_solution(centroids, scale=2.0)
        model.apply_bias_offset(delta=0.3, class_idx=0)

        res = simulate_dynamics(
            model, X, y, learning_rate=0.005, num_steps=50, adapt_mode="bias_only", objective="tent"
        )

        cond_ents = res["cond_entropy"]
        # Check that loss at step t+1 <= loss at step t + small numerical tolerance
        diffs = np.diff(cond_ents)
        assert (diffs <= 1e-5).all(), f"Entropy increased during small-step GD: max diff = {diffs.max()}"

    def test_identical_seed_identical_trajectory(self) -> None:
        """Verify exact determinism: identical seed produces identical trajectory arrays."""
        def run_sim(seed: int) -> np.ndarray:
            X, y, centroids = generate_gaussian_mixture(
                num_classes=3, dim=2, samples_per_class=200, separation=2.0, noise_std=1.0, seed=seed
            )
            model = LinearSoftmaxModel(num_classes=3, dim=2)
            model.fit_centroid_solution(centroids, scale=2.0)
            model.apply_bias_offset(delta=0.8, class_idx=0)
            res = simulate_dynamics(
                model, X, y, learning_rate=0.05, num_steps=60, adapt_mode="bias_only", objective="tent"
            )
            return res["marginal_entropy"]

        traj1 = run_sim(seed=1234)
        traj2 = run_sim(seed=1234)
        np.testing.assert_allclose(traj1, traj2, atol=1e-7)
