"""Local stability tools for entropy minimisation and information maximisation.

Model
-----
For inputs ``x_i``, parameters ``W`` and ``b``, and ``K`` classes,

    z_i = W x_i + b
    p_i = softmax(z_i)
    p_bar = (1/N) sum_i p_i

The two objectives are

    L_ent  = (1/N) sum_i H(p_i)
    L_info = L_ent - lambda H(p_bar)

This module provides analytic gradients with respect to the logits and the
parameters, an autograd Hessian helper, and the reduced ``W = 0`` bias flow.
The theoretical derivation and its scope are documented in ``THEORY.md``.
All Phase 2 numerical checks use float64.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from metrics.entropy import marginal_entropy as _marginal_entropy
from metrics.entropy import predictive_entropy as _predictive_entropy

_DEFAULT_DTYPE = torch.float64


@dataclass(frozen=True)
class StabilitySpec:
    """Validated sizes and lambda values for a Phase 2 stability check."""

    num_classes: int
    dim: int
    num_samples: int
    lambda_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if self.dim < 1:
            raise ValueError("dim must be >= 1")
        if self.num_samples < 2:
            raise ValueError("num_samples must be >= 2")
        if not self.lambda_values:
            raise ValueError("lambda_values must not be empty")
        for value in self.lambda_values:
            validate_lambda(value)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> StabilitySpec:
        """Build a spec from the ``stability`` config section."""
        values = raw.get("lambda_values")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError("stability.lambda_values must be a sequence")
        return cls(
            num_classes=int(raw.get("num_classes", 3)),
            dim=int(raw.get("dim", 2)),
            num_samples=int(raw.get("num_samples", 6)),
            lambda_values=tuple(float(value) for value in values),
        )


def validate_lambda(value: float) -> float:
    """Validate a non-negative, finite information-weight ``lambda``."""
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("lambda must be finite and >= 0")
    return value


def validate_logits(logits: Tensor) -> Tensor:
    """Validate a floating-point ``[N, K]`` logits tensor."""
    if not isinstance(logits, Tensor):
        raise TypeError(f"logits must be a Tensor, got {type(logits)!r}")
    if logits.ndim != 2:
        raise ValueError(f"logits must be [num_samples, num_classes], got {tuple(logits.shape)}")
    if not logits.is_floating_point():
        raise TypeError(f"logits must be floating point, got {logits.dtype}")
    if logits.numel() == 0:
        raise ValueError("logits must not be empty")
    return logits


def softmax(logits: Tensor) -> Tensor:
    """Softmax over the class dimension."""
    return torch.softmax(validate_logits(logits), dim=-1)


def mean_predictive_entropy(logits: Tensor) -> Tensor:
    """Mean per-sample entropy ``(1/N) sum_i H(p_i)``."""
    return _predictive_entropy(validate_logits(logits), reduction="mean")


def marginal_entropy(logits: Tensor) -> Tensor:
    """Marginal entropy ``H(p_bar)``."""
    return _marginal_entropy(validate_logits(logits)).mean()


def information_objective(logits: Tensor, lam: float) -> Tensor:
    """Return ``L_info = mean_i H(p_i) - lambda H(p_bar)``."""
    lam = validate_lambda(lam)
    return mean_predictive_entropy(logits) - lam * marginal_entropy(logits)


def _entropy_and_covariance(logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``p``, ``H(p)`` per sample, and ``diag(p) - p p^T`` per sample."""
    logits = validate_logits(logits)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropies = -(probs * log_probs).sum(dim=-1, keepdim=True)
    covariance = torch.diag_embed(probs) - probs.unsqueeze(-1) * probs.unsqueeze(-2)
    return log_probs, entropies, covariance


def entropy_logit_gradient(logits: Tensor) -> Tensor:
    """Gradient of ``mean_i H(p_i)`` with respect to each logit vector.

    Derived result:

        d L_ent / d z_i = -(1/N) C_i (log p_i + H(p_i) 1)

    where ``C_i = diag(p_i) - p_i p_i^T``.  The gradient is orthogonal to the
    all-ones direction, as required by softmax shift invariance.
    """
    log_probs, entropies, covariance = _entropy_and_covariance(logits)
    inner = log_probs + entropies
    return -(covariance @ inner.unsqueeze(-1)).squeeze(-1) / logits.shape[0]


def information_logit_gradient(logits: Tensor, lam: float) -> Tensor:
    """Gradient of ``L_info`` with respect to each logit vector.

    Derived result:

        d L_info / d z_i
            = -(1/N) C_i (log p_i + H(p_i) 1 - lambda log p_bar)

    with ``C_i = diag(p_i) - p_i p_i^T`` and ``p_bar = mean_i p_i``.
    """
    lam = validate_lambda(lam)
    log_probs, entropies, covariance = _entropy_and_covariance(logits)
    p_bar = log_probs.exp().mean(dim=0)
    log_p_bar = torch.log(p_bar)
    inner = log_probs + entropies - lam * log_p_bar.unsqueeze(0)
    return -(covariance @ inner.unsqueeze(-1)).squeeze(-1) / logits.shape[0]


def parameter_gradients(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """Return ``(dL/dW, dL/db)`` for ``L_info`` using the analytic logit gradient.

    ``x`` has shape ``[N, d]``, ``weight`` ``[K, d]``, and ``bias`` ``[K]``.
    The returned gradients are the exact chain-rule results, not autograd
    outputs.  They are compared with autograd in the Phase 2 script and tests.
    """
    logits = x @ weight.T + bias
    logit_gradient = information_logit_gradient(logits, lam)
    return logit_gradient.T @ x, logit_gradient.sum(dim=0)


def parameter_hessian(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    lam: float,
) -> Tensor:
    """Autograd Hessian of ``L_info`` with respect to flattened ``(W, b)``.

    The parameters must be floating-point leaf tensors with
    ``requires_grad=True``.  The returned matrix has shape
    ``[K*d + K, K*d + K]`` and is symmetric up to floating-point error.
    """
    if not weight.requires_grad or not bias.requires_grad:
        raise ValueError("weight and bias must require gradients")
    if not x.is_floating_point() or not weight.is_floating_point() or not bias.is_floating_point():
        raise TypeError("x, weight, and bias must be floating point")
    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("expected x [N, d], weight [K, d], bias [K]")
    if x.shape[1] != weight.shape[1] or weight.shape[0] != bias.shape[0]:
        raise ValueError("incompatible shapes for x, weight, and bias")

    params = [weight, bias]
    loss = information_objective(x @ weight.T + bias, lam)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat = torch.cat([grad.reshape(-1) for grad in grads])
    rows: list[Tensor] = []
    for index in range(flat.numel()):
        row = torch.autograd.grad(
            flat[index],
            params,
            retain_graph=True,
            allow_unused=True,
        )
        rows.append(
            torch.cat(
                [
                    (value if value is not None else torch.zeros_like(param)).reshape(-1)
                    for value, param in zip(row, params)
                ]
            )
        )
    return torch.stack(rows)


def symmetric_b_block_hessian(lam: float, num_classes: int) -> Tensor:
    """Analytic Hessian of ``L_info`` with respect to ``b`` at ``W=0, b=0``.

    At the symmetric point ``p_i = 1/K`` for every sample, so

        d^2 L_info / db^2 = (lambda - 1) C,
        C = diag(1/K) - (1/K^2) 1 1^T.

    The non-zero eigenvalues on the sum-zero subspace are
    ``(lambda - 1) / K``, each with multiplicity ``K - 1``.
    """
    lam = validate_lambda(lam)
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    covariance = torch.eye(num_classes, dtype=_DEFAULT_DTYPE) / num_classes
    covariance = covariance - torch.ones(
        num_classes, num_classes, dtype=_DEFAULT_DTYPE
    ) / (num_classes**2)
    return (lam - 1.0) * covariance


def marginal_mode_curvature(lam: float, num_classes: int) -> float:
    """Return the analytic marginal-mode eigenvalue ``(lambda - 1) / K``."""
    lam = validate_lambda(lam)
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    return (lam - 1.0) / num_classes


def discrete_step_bound(lam: float, num_classes: int) -> float | None:
    """Largest stable step size for the marginal mode at the symmetric point.

    For ``lambda > 1`` the gradient-descent Jacobian eigenvalue is
    ``1 - eta (lambda - 1) / K``, so stability requires

        ``0 < eta < 2 K / (lambda - 1)``.

    For ``lambda <= 1`` the marginal mode is unstable or neutral and no
    positive stable bound exists; the function returns ``None``.
    """
    lam = validate_lambda(lam)
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if lam <= 1.0:
        return None
    return 2.0 * num_classes / (lam - 1.0)


def reduced_bias_gradient(bias: Tensor, lam: float) -> Tensor:
    """Gradient-flow right-hand side for the reduced ``W = 0`` model.

    With ``W = 0``, every sample has the same logits ``b``, so
    ``L_info = (1 - lambda) H(softmax(b))`` and

        db/dt = -dL_info/db
              = (1 - lambda) p * (b - (p . b) 1),   p = softmax(b).

    The right-hand side sums to zero, so a sum-zero initial bias stays
    sum-zero (up to floating-point error).  This is the exact bias-only
    dynamics; it is not the full Phase 1 model, which also adapts ``W``.
    """
    lam = validate_lambda(lam)
    if not isinstance(bias, Tensor):
        raise TypeError("bias must be a Tensor")
    if bias.ndim != 1:
        raise ValueError(f"bias must be [num_classes], got {tuple(bias.shape)}")
    if not bias.is_floating_point():
        raise TypeError("bias must be floating point")
    probs = torch.softmax(bias, dim=-1)
    weighted_mean = (probs * bias).sum()
    return (1.0 - lam) * probs * (bias - weighted_mean)


def reduced_bias_hessian(lam: float, num_classes: int) -> Tensor:
    """Analytic Hessian of the reduced ``W = 0`` objective at ``b = 0``.

    ``L_info = (1 - lambda) H(softmax(b))`` and the Hessian at uniform is
    ``(lambda - 1) C`` with ``C = diag(1/K) - (1/K^2) 1 1^T``.
    """
    return symmetric_b_block_hessian(lam, num_classes)


def simulate_reduced_bias(
    *,
    lam: float,
    initial_bias: Tensor,
    steps: int,
    learning_rate: float,
) -> tuple[dict[str, Any], ...]:
    """Simulate the exact reduced ``W = 0`` gradient descent.

    The update is ``b <- b + eta * reduced_bias_gradient(b, lam)``.  It is
    intentionally the same explicit-Euler scheme used by the full toy model,
    so the measured threshold and step-size bound refer to discrete updates,
    not only to the continuous-time flow.
    """
    lam = validate_lambda(lam)
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and > 0")
    if not isinstance(initial_bias, Tensor) or initial_bias.ndim != 1:
        raise ValueError("initial_bias must be a [num_classes] Tensor")
    if not initial_bias.is_floating_point():
        raise TypeError("initial_bias must be floating point")

    bias = initial_bias.detach().clone()
    records: list[dict[str, Any]] = []
    for step in range(steps + 1):
        probs = torch.softmax(bias, dim=-1)
        entropy = -(probs * torch.log(probs)).sum()
        spread = torch.linalg.vector_norm(bias - bias.mean())
        records.append(
            {
                "step": step,
                "marginal_entropy": float(entropy),
                "bias_spread": float(spread),
                "bias_max_abs": float(bias.abs().max()),
            }
        )
        if step == steps:
            break
        bias = bias + learning_rate * reduced_bias_gradient(bias, lam)

    return tuple(records)


__all__ = [
    "StabilitySpec",
    "discrete_step_bound",
    "entropy_logit_gradient",
    "information_logit_gradient",
    "information_objective",
    "marginal_entropy",
    "marginal_mode_curvature",
    "mean_predictive_entropy",
    "parameter_gradients",
    "parameter_hessian",
    "reduced_bias_gradient",
    "reduced_bias_hessian",
    "simulate_reduced_bias",
    "softmax",
    "symmetric_b_block_hessian",
    "validate_lambda",
    "validate_logits",
]
