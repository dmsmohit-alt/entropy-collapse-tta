"""Numerical correctness tests for entropy metrics."""

from __future__ import annotations

import math

import pytest
import torch

from metrics.entropy import (
    entropy_from_probs,
    entropy_gap,
    marginal_entropy,
    marginal_probabilities,
    predictive_entropy,
    top_class_fraction,
)


def test_uniform_logits_have_maximum_entropy() -> None:
    logits = torch.zeros(5, 3)

    assert torch.allclose(predictive_entropy(logits), torch.tensor(math.log(3.0)), atol=1e-6)
    assert torch.allclose(marginal_entropy(logits), torch.tensor(math.log(3.0)), atol=1e-6)
    assert torch.allclose(entropy_gap(logits), torch.tensor(0.0), atol=1e-6)


def test_confident_disagreeing_predictions_separate_marginal_and_mean() -> None:
    # Each row is almost one-hot, but the batch marginals are balanced.
    logits = torch.tensor([[40.0, -40.0], [-40.0, 40.0]])

    per_sample = predictive_entropy(logits, reduction="none")
    marginal = marginal_entropy(logits)
    gap = entropy_gap(logits)

    assert per_sample.shape == (2,)
    assert torch.all(per_sample < 1e-6)
    assert torch.allclose(marginal, torch.tensor(math.log(2.0)), atol=1e-6)
    assert torch.allclose(gap, torch.tensor(math.log(2.0)), atol=1e-6)
    assert torch.allclose(top_class_fraction(logits), torch.tensor(0.5), atol=1e-6)


def test_entropy_identity_matches_direct_mutual_information() -> None:
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(11, 4, generator=generator)
    probs = torch.softmax(logits, dim=-1)
    batch_size = logits.shape[0]

    joint = probs / batch_size
    p_y = joint.sum(dim=0)
    marginal_h = -(torch.special.xlogy(p_y, p_y)).sum()
    conditional_h = predictive_entropy(logits)
    direct_mi = (
        joint * (joint.log() - p_y.log().unsqueeze(0) - math.log(1.0 / batch_size))
    ).sum()

    assert torch.allclose(marginal_h, marginal_entropy(logits), atol=1e-6)
    assert torch.allclose(marginal_h - conditional_h, direct_mi, atol=1e-5)
    assert torch.allclose(entropy_gap(logits), direct_mi, atol=1e-5)
    assert entropy_gap(logits) >= -1e-6


def test_marginal_entropy_equals_predictive_entropy_for_identical_rows() -> None:
    generator = torch.Generator().manual_seed(3)
    one_row = torch.randn(4, generator=generator)
    logits = one_row.repeat(9, 1)

    assert torch.allclose(
        marginal_entropy(logits),
        predictive_entropy(one_row.unsqueeze(0)),
        atol=1e-6,
    )


def test_softmax_shift_does_not_change_entropy() -> None:
    generator = torch.Generator().manual_seed(5)
    logits = torch.randn(8, 5, generator=generator)

    assert torch.allclose(
        predictive_entropy(logits),
        predictive_entropy(logits + 100.0),
        atol=1e-6,
    )
    assert torch.allclose(
        marginal_entropy(logits),
        marginal_entropy(logits + 100.0),
        atol=1e-6,
    )


def test_probability_entropy_handles_exact_zeros() -> None:
    probs = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])

    entropies = entropy_from_probs(probs, reduction="none")

    assert torch.isfinite(entropies).all()
    assert torch.allclose(entropies, torch.tensor([0.0, math.log(2.0)]), atol=1e-6)
    assert torch.allclose(entropy_from_probs(probs, reduction="sum"), entropies.sum())
    assert torch.allclose(entropy_from_probs(probs, reduction="mean"), entropies.mean())


def test_marginal_probabilities_are_valid_and_top_class_fraction_ranges() -> None:
    logits = torch.tensor(
        [
            [9.0, 0.0, -1.0],
            [8.0, 0.0, -1.0],
            [0.0, 9.0, -1.0],
            [0.0, 8.0, -1.0],
        ]
    )

    p_bar = marginal_probabilities(logits)
    assert p_bar.shape == (3,)
    assert torch.allclose(p_bar.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(top_class_fraction(logits), torch.tensor(0.5), atol=1e-6)
    assert top_class_fraction(torch.tensor([[1.0, 0.0], [2.0, 0.0]])) == 1.0


def test_entropy_gradients_are_finite() -> None:
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(6, 4, generator=generator, requires_grad=True)

    loss = predictive_entropy(logits)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_invalid_inputs_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="shape"):
        predictive_entropy(torch.randn(3))
    with pytest.raises(TypeError, match="floating point"):
        predictive_entropy(torch.ones(3, 2, dtype=torch.int64))
    with pytest.raises(ValueError, match="reduction"):
        predictive_entropy(torch.randn(3, 2), reduction="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sample_dim"):
        marginal_probabilities(torch.randn(3, 2), sample_dim=-1)
    with pytest.raises(ValueError, match="batch, classes"):
        entropy_gap(torch.randn(2, 3, 4))
    with pytest.raises(ValueError, match="empty"):
        top_class_fraction(torch.empty(0, 3))
    with pytest.raises(ValueError, match="empty"):
        entropy_from_probs(torch.empty(0, 3))
