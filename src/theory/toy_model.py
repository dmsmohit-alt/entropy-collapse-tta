"""Toy linear-softmax model on a balanced Gaussian mixture.

Modelling choice for Phase 1
----------------------------
The classifier is ``softmax(W x + b)``.  Both ``W`` and ``b`` are adapted by
full-batch gradient descent on mean predictive entropy.  ``W`` is initialised
at the class-prototype matrix, which gives a high-accuracy source classifier;
``b`` is initialised with a bias towards one class.  The bias magnitude is the
parameter scanned to locate the collapse basin boundary.

Adapting ``W`` matters: if ``W`` is frozen, the balanced solution is not a
stable non-collapsed attractor, and every run eventually collapses.  With
``W`` free, the model can instead sharpen a class-separating solution and keep
the marginal distribution balanced.

The toy model is transductive: unlabeled training samples are used for
adaptation, while a separate held-out sample is used for all reported metrics.
Labels are used only for diagnostics, never in the objective.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from metrics.collapse import CollapseCriteria, CollapseDetector
from metrics.entropy import (
    entropy_gap,
    marginal_entropy,
    predictive_entropy,
    softmax,
    top_class_fraction,
)


@dataclass(frozen=True)
class ToySetup:
    """Static configuration shared by every trajectory in one experiment."""

    num_classes: int = 3
    dim: int = 3
    separation: float = 3.0
    train_per_class: int = 600
    eval_per_class: int = 1000
    steps: int = 2500
    learning_rate: float = 0.03
    eval_every: int = 10
    initial_bias_class: int = 0
    collapse: CollapseCriteria = CollapseCriteria()

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if self.dim < self.num_classes:
            raise ValueError("dim must be >= num_classes for prototype means")
        if not math.isfinite(self.separation) or self.separation <= 0:
            raise ValueError("separation must be finite and > 0")
        if self.train_per_class < 1 or self.eval_per_class < 1:
            raise ValueError("train_per_class and eval_per_class must be >= 1")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and > 0")
        if self.eval_every < 1:
            raise ValueError("eval_every must be >= 1")
        if not 0 <= self.initial_bias_class < self.num_classes:
            raise ValueError("initial_bias_class must index a class")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ToySetup:
        """Build and validate a setup from the ``toy`` config section."""
        collapse_raw = raw.get("collapse", {})
        if not isinstance(collapse_raw, Mapping):
            raise ValueError("toy.collapse must be a mapping")
        collapse = CollapseCriteria(
            marginal_entropy_threshold=float(collapse_raw.get("marginal_entropy", 0.10)),
            top_class_fraction_threshold=float(collapse_raw.get("top_class_fraction", 0.95)),
            persistence=int(collapse_raw.get("persistence", 20)),
        )
        return cls(
            num_classes=int(raw.get("num_classes", 3)),
            dim=int(raw.get("dim", 3)),
            separation=float(raw.get("separation", 3.0)),
            train_per_class=int(raw.get("train_per_class", 600)),
            eval_per_class=int(raw.get("eval_per_class", 1000)),
            steps=int(raw.get("steps", 2500)),
            learning_rate=float(raw.get("learning_rate", 0.03)),
            eval_every=int(raw.get("eval_every", 10)),
            initial_bias_class=int(raw.get("initial_bias_class", 0)),
            collapse=collapse,
        )


@dataclass(frozen=True)
class MixtureData:
    """Balanced train/eval splits and the class-mean matrix."""

    x_train: Tensor
    y_train: Tensor
    x_eval: Tensor
    y_eval: Tensor
    means: Tensor


@dataclass(frozen=True)
class TrajectoryResult:
    """One entropy-minimisation trajectory."""

    records: tuple[dict[str, Any], ...]
    collapse_step: int | None
    onset_step: int | None
    final: dict[str, Any]


def make_gaussian_mixture(setup: ToySetup, *, seed: int) -> MixtureData:
    """Generate balanced Gaussian classes with means ``separation * e_k``.

    Train and evaluation samples are drawn independently.  Shuffling is done
    with the same local generator so the data order is reproducible.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")

    generator = torch.Generator().manual_seed(seed)
    means = torch.zeros(setup.num_classes, setup.dim)
    means[:, : setup.num_classes] = torch.eye(setup.num_classes) * setup.separation

    def split(per_class: int) -> tuple[Tensor, Tensor]:
        labels = torch.arange(setup.num_classes).repeat_interleave(per_class)
        x = means[labels] + torch.randn(labels.numel(), setup.dim, generator=generator)
        order = torch.randperm(labels.numel(), generator=generator)
        return x[order], labels[order]

    x_train, y_train = split(setup.train_per_class)
    x_eval, y_eval = split(setup.eval_per_class)
    return MixtureData(
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        y_eval=y_eval,
        means=means,
    )


def initial_parameters(
    setup: ToySetup,
    *,
    initial_bias: float,
) -> tuple[Tensor, Tensor]:
    """Return trainable ``W`` and ``b`` with a bias towards one class.

    ``W`` is the class-prototype matrix restricted to the first ``K``
    coordinates.  This is a deliberately strong but imperfect source model.
    """
    if not math.isfinite(initial_bias):
        raise ValueError("initial_bias must be finite")
    weight = torch.zeros(setup.num_classes, setup.dim)
    weight[:, : setup.num_classes] = torch.eye(setup.num_classes)
    bias = torch.zeros(setup.num_classes)
    bias[setup.initial_bias_class] = initial_bias
    return weight.requires_grad_(True), bias.requires_grad_(True)


def evaluate_logits(logits: Tensor, labels: Tensor) -> dict[str, float]:
    """Compute evaluation diagnostics from logits and held-out labels."""
    if logits.ndim != 2:
        raise ValueError("logits must be [batch, classes]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have one entry per logits row")

    with torch.no_grad():
        return {
            "mean_predictive_entropy": float(predictive_entropy(logits)),
            "marginal_entropy": float(marginal_entropy(logits)),
            "entropy_gap": float(entropy_gap(logits)),
            "top_class_fraction": float(top_class_fraction(logits)),
            "accuracy": float((logits.argmax(dim=-1) == labels).float().mean()),
            "mean_max_probability": float(softmax(logits).max(dim=-1).values.mean()),
        }


def source_metrics(data: MixtureData, setup: ToySetup) -> dict[str, float]:
    """Metrics of the unadapted class-prototype model ``W=I, b=0``."""
    weight = torch.zeros(setup.num_classes, setup.dim)
    weight[:, : setup.num_classes] = torch.eye(setup.num_classes)
    bias = torch.zeros(setup.num_classes)
    return evaluate_logits(data.x_eval @ weight.T + bias, data.y_eval)


def _snapshot(
    step: int,
    weight: Tensor,
    bias: Tensor,
    data: MixtureData,
    *,
    detector: CollapseDetector,
    collapse_event: bool,
) -> dict[str, Any]:
    with torch.no_grad():
        eval_metrics = evaluate_logits(data.x_eval @ weight.T + bias, data.y_eval)
        train_loss = float(predictive_entropy(data.x_train @ weight.T + bias))
        bias_values = bias.detach().tolist()
        record: dict[str, Any] = {
            "step": step,
            "train_loss": train_loss,
            "eval_mean_predictive_entropy": eval_metrics["mean_predictive_entropy"],
            "eval_marginal_entropy": eval_metrics["marginal_entropy"],
            "eval_entropy_gap": eval_metrics["entropy_gap"],
            "eval_top_class_fraction": eval_metrics["top_class_fraction"],
            "eval_accuracy": eval_metrics["accuracy"],
            "eval_mean_max_probability": eval_metrics["mean_max_probability"],
            "bias_max_minus_mean": float(bias.max() - bias.mean()),
            "weight_frobenius": float(weight.norm()),
            "bias": bias_values,
            "collapse_event": collapse_event,
            "collapsed": detector.is_collapsed,
        }
        for index, value in enumerate(bias_values):
            record[f"bias_{index}"] = float(value)
        return record


def simulate_entropy_minimisation(
    data: MixtureData,
    setup: ToySetup,
    *,
    initial_bias: float,
) -> TrajectoryResult:
    """Run full-batch gradient descent on mean predictive entropy.

    The objective is ``mean_i H(p_theta(. | x_i))`` on unlabeled training
    samples.  No labels, marginal-entropy term, weight decay, or reset is used
    in Phase 1.  Evaluation happens every ``eval_every`` steps and always at
    the final step, including step zero before any update.
    """
    weight, bias = initial_parameters(setup, initial_bias=initial_bias)
    optimizer = torch.optim.SGD([weight, bias], lr=setup.learning_rate)
    detector = CollapseDetector(setup.collapse)

    records: list[dict[str, Any]] = []
    initial = _snapshot(
        0,
        weight,
        bias,
        data,
        detector=detector,
        collapse_event=False,
    )
    records.append(initial)

    for step in range(1, setup.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = data.x_train @ weight.T + bias
        loss = predictive_entropy(logits)
        loss.backward()
        optimizer.step()

        if step % setup.eval_every != 0 and step != setup.steps:
            continue

        # Evaluate first, then update the detector, so the logged metrics are
        # exactly the metrics on which the collapse decision is based.
        with torch.no_grad():
            eval_logits = data.x_eval @ weight.T + bias
            eval_metrics = evaluate_logits(eval_logits, data.y_eval)
        event = detector.update(
            step,
            eval_metrics["marginal_entropy"],
            eval_metrics["top_class_fraction"],
        )
        records.append(
            _snapshot(
                step,
                weight,
                bias,
                data,
                detector=detector,
                collapse_event=event,
            )
        )

    final = {key: value for key, value in records[-1].items() if key not in {"bias", "collapse_event"}}
    final["bias"] = list(records[-1]["bias"])
    return TrajectoryResult(
        records=tuple(records),
        collapse_step=detector.collapsed_at,
        onset_step=detector.onset_step,
        final=final,
    )


__all__ = [
    "MixtureData",
    "ToySetup",
    "TrajectoryResult",
    "evaluate_logits",
    "initial_parameters",
    "make_gaussian_mixture",
    "simulate_entropy_minimisation",
    "source_metrics",
]
