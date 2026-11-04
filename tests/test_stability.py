"""Tests for the Phase 2 analytic gradients and stability predictions."""

from __future__ import annotations

import pytest
import torch

from theory.stability import (
    discrete_step_bound,
    entropy_logit_gradient,
    information_logit_gradient,
    information_objective,
    marginal_mode_curvature,
    mean_predictive_entropy,
    parameter_gradients,
    parameter_hessian,
    reduced_bias_gradient,
    simulate_reduced_bias,
    symmetric_b_block_hessian,
)


def _logits(seed: int = 0, n: int = 5, k: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, k, generator=generator, dtype=torch.float64)


def test_entropy_logit_gradient_matches_autograd() -> None:
    logits = _logits().requires_grad_(True)
    autograd_grad = torch.autograd.grad(mean_predictive_entropy(logits), logits)[0]
    analytic = entropy_logit_gradient(logits.detach())

    assert torch.allclose(autograd_grad, analytic, atol=1e-12, rtol=1e-10)
    # Softmax shift invariance: the gradient sums to zero over classes.
    assert torch.allclose(analytic.sum(dim=-1), torch.zeros(5, dtype=torch.float64))


def test_information_logit_gradient_matches_autograd() -> None:
    logits = _logits(seed=1).requires_grad_(True)
    lam = 1.7
    autograd_grad = torch.autograd.grad(information_objective(logits, lam), logits)[0]
    analytic = information_logit_gradient(logits.detach(), lam)

    assert torch.allclose(autograd_grad, analytic, atol=1e-12, rtol=1e-10)
    assert torch.allclose(analytic.sum(dim=-1), torch.zeros(5, dtype=torch.float64))


def test_parameter_gradients_match_autograd() -> None:
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(5, 2, generator=generator, dtype=torch.float64)
    weight = torch.randn(3, 2, generator=generator, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(3, generator=generator, dtype=torch.float64, requires_grad=True)
    lam = 0.8

    loss = information_objective(x @ weight.T + bias, lam)
    autograd_weight, autograd_bias = torch.autograd.grad(loss, [weight, bias])
    analytic_weight, analytic_bias = parameter_gradients(x, weight.detach(), bias.detach(), lam)

    assert torch.allclose(autograd_weight, analytic_weight, atol=1e-12, rtol=1e-10)
    assert torch.allclose(autograd_bias, analytic_bias, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("lam", [0.5, 2.0])
def test_symmetric_b_block_matches_analytic_prediction(lam: float) -> None:
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(6, 2, generator=generator, dtype=torch.float64)
    weight = torch.zeros(3, 2, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(3, dtype=torch.float64, requires_grad=True)

    hessian = parameter_hessian(x, weight, bias, lam)
    b_block = hessian[6:, 6:]
    expected = symmetric_b_block_hessian(lam, 3)

    assert torch.allclose(b_block, expected, atol=1e-12, rtol=1e-10)
    eigenvalues = torch.linalg.eigvalsh(b_block)
    predicted = torch.sort(
        torch.tensor(
            [(lam - 1.0) / 3.0, (lam - 1.0) / 3.0, 0.0],
            dtype=torch.float64,
        )
    ).values
    assert torch.allclose(eigenvalues, predicted, atol=1e-10, rtol=1e-9)


def test_centered_features_leave_only_the_marginal_mode_positive() -> None:
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(6, 2, generator=generator, dtype=torch.float64)
    x = x - x.mean(dim=0, keepdim=True)
    weight = torch.zeros(3, 2, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    lam = 2.0

    hessian = parameter_hessian(x, weight, bias, lam)
    eigenvalues = torch.linalg.eigvalsh(hessian)

    assert eigenvalues.max().item() == pytest.approx(marginal_mode_curvature(lam, 3), abs=1e-9)
    assert eigenvalues.min().item() <= 0.0


def test_reduced_bias_gradient_is_sum_zero_and_has_correct_sign() -> None:
    bias = torch.tensor([0.4, -0.1, -0.3], dtype=torch.float64)
    for lam in (0.0, 0.5, 1.0, 2.0):
        gradient = reduced_bias_gradient(bias, lam)
        assert gradient.sum().item() == pytest.approx(0.0, abs=1e-12)
    assert (reduced_bias_gradient(bias, 0.5) * bias).sum().item() > 0.0
    assert (reduced_bias_gradient(bias, 2.0) * bias).sum().item() < 0.0


def test_reduced_dynamics_contracts_above_one_and_expands_below_one() -> None:
    initial = 0.01 * torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64)

    stable = simulate_reduced_bias(
        lam=2.0,
        initial_bias=initial,
        steps=300,
        learning_rate=0.1,
    )
    unstable = simulate_reduced_bias(
        lam=0.5,
        initial_bias=initial,
        steps=300,
        learning_rate=0.1,
    )
    neutral = simulate_reduced_bias(
        lam=1.0,
        initial_bias=initial,
        steps=300,
        learning_rate=0.1,
    )

    assert stable[-1]["bias_spread"] < stable[0]["bias_spread"]
    assert stable[-1]["marginal_entropy"] > stable[0]["marginal_entropy"]
    assert unstable[-1]["bias_spread"] > unstable[0]["bias_spread"]
    assert unstable[-1]["marginal_entropy"] < unstable[0]["marginal_entropy"]
    assert neutral[-1]["bias_spread"] == pytest.approx(neutral[0]["bias_spread"], abs=1e-15)


@pytest.mark.parametrize("learning_rate", [1.0, 3.0, 5.8])
def test_one_step_factor_below_the_bound_contracts(learning_rate: float) -> None:
    lam = 2.0
    initial = 1e-6 * torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64)
    records = simulate_reduced_bias(
        lam=lam,
        initial_bias=initial,
        steps=1,
        learning_rate=learning_rate,
    )
    observed = records[1]["bias_spread"] / records[0]["bias_spread"]
    predicted = 1.0 - learning_rate * (lam - 1.0) / 3.0

    assert observed == pytest.approx(abs(predicted), abs=1e-4)
    assert abs(predicted) < 1.0


def test_one_step_factor_above_the_bound_expands() -> None:
    lam = 2.0
    learning_rate = 6.2
    initial = 1e-6 * torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64)
    records = simulate_reduced_bias(
        lam=lam,
        initial_bias=initial,
        steps=1,
        learning_rate=learning_rate,
    )
    observed = records[1]["bias_spread"] / records[0]["bias_spread"]
    predicted = 1.0 - learning_rate * (lam - 1.0) / 3.0

    assert observed == pytest.approx(abs(predicted), rel=1e-6)
    assert abs(predicted) > 1.0


def test_step_bound_and_marginal_curvature_formulas() -> None:
    assert marginal_mode_curvature(2.0, 3) == pytest.approx(1.0 / 3.0)
    assert discrete_step_bound(2.0, 3) == pytest.approx(6.0)
    assert discrete_step_bound(1.0, 3) is None
    assert discrete_step_bound(0.5, 3) is None

    with pytest.raises(ValueError):
        marginal_mode_curvature(2.0, 1)
    with pytest.raises(ValueError):
        discrete_step_bound(-0.1, 3)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        entropy_logit_gradient(torch.randn(3))
    with pytest.raises(ValueError):
        information_logit_gradient(torch.randn(3, 3, dtype=torch.float64), -1.0)
    with pytest.raises(ValueError):
        reduced_bias_gradient(torch.randn(2, 2, dtype=torch.float64), 1.0)
    with pytest.raises(ValueError):
        simulate_reduced_bias(
            lam=2.0,
            initial_bias=torch.zeros(3, dtype=torch.float64),
            steps=0,
            learning_rate=0.1,
        )
