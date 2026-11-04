"""Tests for the synthetic mixture and toy entropy-minimisation dynamics."""

from __future__ import annotations

import pytest
import torch

from theory.toy_model import (
    ToySetup,
    evaluate_logits,
    initial_parameters,
    make_gaussian_mixture,
    simulate_entropy_minimisation,
    source_metrics,
)


def _setup() -> ToySetup:
    return ToySetup(
        num_classes=3,
        dim=3,
        separation=3.0,
        train_per_class=20,
        eval_per_class=30,
        steps=40,
        learning_rate=0.03,
        eval_every=10,
    )


def test_mixture_is_balanced_and_reproducible() -> None:
    setup = _setup()
    first = make_gaussian_mixture(setup, seed=7)
    second = make_gaussian_mixture(setup, seed=7)
    other = make_gaussian_mixture(setup, seed=8)

    assert first.x_train.shape == (60, 3)
    assert first.x_eval.shape == (90, 3)
    assert torch.equal(first.y_train.bincount(), torch.tensor([20, 20, 20]))
    assert torch.equal(first.y_eval.bincount(), torch.tensor([30, 30, 30]))
    assert torch.equal(first.x_train, second.x_train)
    assert not torch.equal(first.x_train, other.x_train)
    assert torch.allclose(first.means[:, :3], torch.eye(3) * 3.0)


def test_initial_model_shape_and_bias() -> None:
    setup = _setup()
    weight, bias = initial_parameters(setup, initial_bias=1.25)

    assert weight.shape == (3, 3)
    assert weight.requires_grad and bias.requires_grad
    assert torch.allclose(weight, torch.eye(3))
    assert torch.allclose(bias, torch.tensor([1.25, 0.0, 0.0]))


def test_source_classifier_is_strong_on_separated_mixture() -> None:
    setup = ToySetup(
        num_classes=3,
        dim=3,
        separation=3.0,
        train_per_class=20,
        eval_per_class=100,
    )
    data = make_gaussian_mixture(setup, seed=0)
    metrics = source_metrics(data, setup)

    assert metrics["accuracy"] > 0.9
    assert metrics["top_class_fraction"] < 0.45


def test_evaluate_logits_reports_expected_quantities() -> None:
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    labels = torch.tensor([0, 1])

    metrics = evaluate_logits(logits, labels)

    assert metrics["accuracy"] == 1.0
    assert metrics["top_class_fraction"] == pytest.approx(0.5)
    assert metrics["marginal_entropy"] == pytest.approx(0.693147, rel=1e-5)
    assert metrics["mean_predictive_entropy"] < 1e-3


def test_simulation_logs_step_zero_and_final_step_and_adapts() -> None:
    setup = _setup()
    data = make_gaussian_mixture(setup, seed=3)

    result = simulate_entropy_minimisation(data, setup, initial_bias=0.5)

    assert result.records[0]["step"] == 0
    assert result.records[-1]["step"] == setup.steps
    assert [record["step"] for record in result.records] == [0, 10, 20, 30, 40]
    assert result.final["eval_accuracy"] >= 0.0
    assert result.final["eval_accuracy"] <= 1.0
    assert result.collapse_step is None
    assert result.records[-1]["weight_frobenius"] > result.records[0]["weight_frobenius"]
    assert len(result.final["bias"]) == 3


def test_simulation_always_evaluates_the_final_step() -> None:
    setup = ToySetup(
        num_classes=3,
        dim=3,
        separation=3.0,
        train_per_class=20,
        eval_per_class=30,
        steps=45,
        learning_rate=0.03,
        eval_every=10,
    )
    data = make_gaussian_mixture(setup, seed=4)

    result = simulate_entropy_minimisation(data, setup, initial_bias=0.5)

    assert [record["step"] for record in result.records] == [0, 10, 20, 30, 40, 45]
    assert result.records[-1]["step"] == setup.steps


def test_setup_rejects_invalid_dimension() -> None:
    with pytest.raises(ValueError, match="dim"):
        ToySetup(num_classes=4, dim=3)
    with pytest.raises(ValueError, match="learning_rate"):
        ToySetup(learning_rate=0)
