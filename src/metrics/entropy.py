"""Entropy metrics for test-time adaptation.

All entropies are measured in nats (natural logarithms).  For a batch of
logits the two central quantities are

    mean predictive entropy = E_x[H(p_theta(. | x))]
    marginal entropy        = H(E_x[p_theta(. | x)]) = H(p_bar)

They are related by the exact identity

    E_x[H(p_theta(. | x))] = H(p_bar) - I(X; Y_theta),

where the empirical mutual-information term is non-negative.  Entropy
minimisation alone can therefore reduce H(p_bar) to zero, which is the
marginal-collapse failure mode studied in this project.

The implementations use ``log_softmax`` and ``torch.special.xlogy`` rather
than explicit ``log(p)`` so that confident predictions do not produce NaNs.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

Reduction = Literal["none", "mean", "sum"]


def _as_logits(logits: Tensor) -> Tensor:
    """Validate a ``[..., K]`` logits tensor and return it unchanged."""
    if not isinstance(logits, Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits)!r}")
    if logits.ndim < 2:
        raise ValueError(f"logits must have shape [..., num_classes], got {tuple(logits.shape)}")
    if not logits.is_floating_point():
        raise TypeError(f"logits must be floating point, got {logits.dtype}")
    if logits.numel() == 0:
        raise ValueError("logits must not be empty")
    return logits


def _reduce(values: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError(f"reduction must be 'none', 'mean', or 'sum'; got {reduction!r}")


def softmax(logits: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable softmax over ``dim``."""
    if not isinstance(logits, Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits)!r}")
    if not logits.is_floating_point():
        raise TypeError(f"logits must be floating point, got {logits.dtype}")
    if logits.numel() == 0:
        raise ValueError("logits must not be empty")
    return torch.softmax(logits, dim=dim)


def entropy_from_probs(
    probs: Tensor,
    *,
    dim: int = -1,
    reduction: Reduction = "none",
) -> Tensor:
    """Shannon entropy of probability vectors along ``dim``.

    ``torch.special.xlogy`` is used because the limiting value
    ``0 * log(0)`` must be treated as zero.  This function does not
    renormalise ``probs``; callers are responsible for passing valid
    probability vectors.
    """
    if not isinstance(probs, Tensor):
        raise TypeError(f"probs must be a torch.Tensor, got {type(probs)!r}")
    if probs.ndim < 1:
        raise ValueError("probs must have at least one dimension")
    if not probs.is_floating_point():
        raise TypeError(f"probs must be floating point, got {probs.dtype}")
    if probs.numel() == 0:
        raise ValueError("probs must not be empty")

    entropy = -torch.special.xlogy(probs, probs).sum(dim=dim)
    return _reduce(entropy, reduction)


def predictive_entropy(
    logits: Tensor,
    *,
    reduction: Reduction = "mean",
) -> Tensor:
    """Entropy of each prediction, optionally reduced over the batch.

    Args:
        logits: Tensor with shape ``[..., num_classes]``.
        reduction: ``"none"`` returns one entropy per input; ``"mean"`` and
            ``"sum"`` reduce over every non-class dimension.
    """
    logits = _as_logits(logits)
    log_probs = torch.log_softmax(logits, dim=-1)
    # log_probs is finite even for extremely confident predictions, so
    # underflowed probabilities contribute exactly zero rather than NaN.
    per_sample = -(log_probs.exp() * log_probs).sum(dim=-1)
    return _reduce(per_sample, reduction)


def marginal_probabilities(logits: Tensor, *, sample_dim: int = 0) -> Tensor:
    """Average predicted probabilities over the sample dimension.

    For the standard TTA case, ``logits`` has shape ``[batch, classes]`` and
    ``sample_dim=0`` returns a single ``[classes]`` vector.
    """
    logits = _as_logits(logits)
    if sample_dim < -logits.ndim or sample_dim >= logits.ndim:
        raise ValueError(f"sample_dim={sample_dim} is invalid for shape {tuple(logits.shape)}")
    if sample_dim == -1 or sample_dim == logits.ndim - 1:
        raise ValueError("sample_dim must not be the class dimension (last dimension)")
    return softmax(logits, dim=-1).mean(dim=sample_dim)


def marginal_entropy(logits: Tensor, *, sample_dim: int = 0) -> Tensor:
    """Entropy of the mean predicted distribution, ``H(p_bar)``.

    The result is a scalar for ``[batch, classes]`` input.  For higher-rank
    input with a non-default ``sample_dim``, leading dimensions are retained.
    """
    p_bar = marginal_probabilities(logits, sample_dim=sample_dim)
    return entropy_from_probs(p_bar, reduction="none")


def entropy_gap(logits: Tensor) -> Tensor:
    """Return ``H(p_bar) - mean_i H(p_i)`` for ``[batch, classes]`` logits.

    This is the finite-batch estimate of the mutual information ``I(X; Y)``.
    It is non-negative up to floating-point error by Jensen's inequality.
    """
    logits = _as_logits(logits)
    if logits.ndim != 2:
        raise ValueError(f"entropy_gap expects [batch, classes] logits, got {tuple(logits.shape)}")
    return marginal_entropy(logits, sample_dim=0) - predictive_entropy(logits, reduction="mean")


def top_class_fraction(logits: Tensor) -> Tensor:
    """Fraction of predictions assigned to the most frequent predicted class.

    This is a batch-level diagnostic.  Later phases use it inside a collapse
    detector with persistence and smoothing; a single high value here is not
    itself defined as collapse.
    """
    logits = _as_logits(logits)
    predictions = logits.argmax(dim=-1)
    counts = torch.bincount(predictions.reshape(-1), minlength=logits.shape[-1])
    return counts.max().to(dtype=logits.dtype) / predictions.numel()


__all__ = [
    "Reduction",
    "entropy_from_probs",
    "entropy_gap",
    "marginal_entropy",
    "marginal_probabilities",
    "predictive_entropy",
    "softmax",
    "top_class_fraction",
]
