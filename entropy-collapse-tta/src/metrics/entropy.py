"""Information-theoretic metrics and objectives for test-time adaptation.

Provides numerically stable implementations of:
- Conditional prediction entropy: E_x[H(p_theta(.|x))]
- Marginal output distribution: p_bar = E_x[p_theta(.|x)]
- Marginal entropy: H(p_bar)
- Information-maximisation (IM) objective: E[H(p)] - lambda * H(p_bar)
- Mutual information estimate: I(X; Y_theta) = H(p_bar) - E[H(p)]
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F


def compute_conditional_entropy(
    logits: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute conditional Shannon entropy H(p(.|x)) in nats.

    Given unnormalised logit predictions z for a batch of inputs,
    computes H(p) = - sum_k p_k * log(p_k).

    Numerical stability:
        Uses log_softmax to prevent overflow, and guards 0 * -inf = NaN
        via torch.where mask.

    Args:
        logits: Tensor of shape (..., K) containing unnormalised scores.
        reduction: "none" (returns (...,)), "mean" (scalar), or "sum" (scalar).

    Returns:
        Entropy tensor matching reduction specification.
    """
    if logits.ndim == 0:
        raise ValueError("Logits must have at least 1 dimension (classes).")

    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    # 0 * log(0) limit is 0; guard against 0.0 * -inf = NaN
    plogp = torch.where(probs > 0.0, probs * log_probs, torch.zeros_like(probs))
    ent = -torch.sum(plogp, dim=-1)

    if reduction == "mean":
        return ent.mean()
    elif reduction == "sum":
        return ent.sum()
    elif reduction == "none":
        return ent
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Choose 'mean', 'sum', or 'none'.")


def compute_marginal_distribution(logits: torch.Tensor) -> torch.Tensor:
    """Compute the empirical marginal prediction distribution p_bar over a batch.

    p_bar_k = (1 / B) * sum_{i=1}^B p(y=k | x_i)

    Args:
        logits: Tensor of shape (B, K) where B is batch size and K is num classes.

    Returns:
        Tensor of shape (K,) summing to 1.0.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits (B, K), got shape {tuple(logits.shape)}")
    probs = F.softmax(logits, dim=-1)
    return probs.mean(dim=0)


def compute_marginal_entropy(
    logits: Optional[torch.Tensor] = None,
    marginal_p: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute the entropy of the marginal class distribution H(p_bar).

    H(p_bar) = - sum_{k=1}^K p_bar_k * log(p_bar_k)

    Either `logits` or `marginal_p` must be provided.

    Args:
        logits: (B, K) batch predictions.
        marginal_p: (K,) precomputed marginal distribution.
        eps: Small constant to prevent log(0).

    Returns:
        Scalar tensor containing H(p_bar) in nats.
    """
    if marginal_p is None:
        if logits is None:
            raise ValueError("Either logits or marginal_p must be provided.")
        marginal_p = compute_marginal_distribution(logits)

    if marginal_p.ndim != 1:
        raise ValueError(f"Expected 1D marginal distribution (K,), got shape {tuple(marginal_p.shape)}")

    # Clamp for numerical safety under extreme collapse
    p_clamped = torch.clamp(marginal_p, min=eps)
    log_p = torch.log(p_clamped)

    ent = -torch.sum(
        torch.where(marginal_p > 0.0, marginal_p * log_p, torch.zeros_like(marginal_p))
    )
    return ent


def compute_im_loss(
    logits: torch.Tensor,
    lambda_im: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the Information Maximisation (IM) loss.

    L_IM = E_x[H(p(y|x))] - lambda_im * H(p_bar)

    Background identity:
        E_x[H(p(y|x))] = H(p_bar) - I(X; Y)
    When lambda_im = 1.0:
        L_IM = - I(X; Y)
    Minimising L_IM simultaneously:
    1. Minimises conditional entropy (encourages sharp, confident predictions)
    2. Maximises marginal entropy (discourages trivial collapse where all inputs
       get assigned to the same dominant class).

    Args:
        logits: (B, K) batch predictions.
        lambda_im: Weight on marginal entropy regulariser.

    Returns:
        loss: Scalar differentiable loss for backprop.
        metrics: Dictionary of diagnostic scalar values for logging.
    """
    cond_ent = compute_conditional_entropy(logits, reduction="mean")
    marg_ent = compute_marginal_entropy(logits=logits)

    loss = cond_ent - lambda_im * marg_ent

    # Mutual information proxy: I(X; Y) = H(p_bar) - E[H(p(y|x))]
    mi = marg_ent - cond_ent

    metrics = {
        "cond_entropy": cond_ent.detach().item(),
        "marginal_entropy": marg_ent.detach().item(),
        "mutual_info": mi.detach().item(),
        "im_loss": loss.detach().item(),
    }
    return loss, metrics


def compute_top_class_fraction(logits: torch.Tensor) -> Tuple[float, int]:
    """Compute fraction of predictions assigned to the most frequent class in batch.

    Used as an early indicator of representation collapse.

    Args:
        logits: (B, K) batch predictions.

    Returns:
        (max_fraction, dominant_class_idx)
    """
    if logits.ndim != 2 or logits.size(0) == 0:
        raise ValueError(f"Expected non-empty 2D logits (B, K), got {logits.shape}")

    preds = torch.argmax(logits, dim=-1)
    batch_size = preds.size(0)
    counts = torch.bincount(preds, minlength=logits.size(-1))
    max_count, dominant_class = torch.max(counts, dim=0)

    fraction = (max_count.float() / float(batch_size)).item()
    return fraction, int(dominant_class.item())
