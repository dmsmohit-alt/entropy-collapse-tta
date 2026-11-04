"""Entropy minimisation dynamics simulation, gradient tracking, and collapse detection.

Simulates gradient descent / gradient flow of entropy minimisation and
information-maximisation on linear-softmax models.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from src.metrics.entropy import (
    compute_conditional_entropy,
    compute_marginal_entropy,
    compute_top_class_fraction,
    compute_im_loss,
)
from src.theory.model import LinearSoftmaxModel


def check_collapse_criteria(
    top_fraction: float,
    marginal_entropy: float,
    num_classes: int,
    definition: str = "standard",
) -> bool:
    """Evaluate whether a model prediction state is collapsed under various criteria.

    Definitions:
        - 'standard': top_fraction >= 0.90 AND marginal_entropy < 0.30 * ln(K)
        - 'def_a' or 'top_frac_0.9': top_fraction >= 0.90
        - 'def_b' or 'h_pbar_0.3': marginal_entropy < 0.30 * ln(K)
        - 'def_c' or 'h_pbar_0.1': marginal_entropy < 0.10 * ln(K)

    Args:
        top_fraction: Dominant class fraction.
        marginal_entropy: Marginal entropy H(p_bar) in nats.
        num_classes: Number of classes K.
        definition: Which collapse rule to evaluate.

    Returns:
        Boolean indicating collapse under chosen definition.
    """
    max_entropy = np.log(num_classes)
    if definition in ("standard", "both"):
        return (top_fraction >= 0.90) and (marginal_entropy < 0.30 * max_entropy)
    elif definition in ("def_a", "top_frac_0.9"):
        return top_fraction >= 0.90
    elif definition in ("def_b", "h_pbar_0.3"):
        return marginal_entropy < 0.30 * max_entropy
    elif definition in ("def_c", "h_pbar_0.1"):
        return marginal_entropy < 0.10 * max_entropy
    else:
        raise ValueError(f"Unknown collapse definition: {definition}")


def is_collapsed_state(
    top_fraction: float,
    marginal_entropy: float,
    num_classes: int,
    top_fraction_threshold: float = 0.90,
    h_pbar_ratio_threshold: float = 0.30,
) -> bool:
    """Standard definition of adaptation collapse (legacy wrapper)."""
    max_entropy = np.log(num_classes)
    h_cutoff = h_pbar_ratio_threshold * max_entropy
    return (top_fraction >= top_fraction_threshold) and (marginal_entropy < h_cutoff)


def simulate_dynamics(
    model: LinearSoftmaxModel,
    X: torch.Tensor,
    y_eval: torch.Tensor,
    learning_rate: float = 0.05,
    num_steps: int = 500,
    adapt_mode: str = "bias_only",
    objective: str = "tent",
    lambda_im: float = 1.0,
    collapse_definition: str = "standard",
    log_every: int = 1,
) -> Dict[str, Any]:
    """Run full-batch gradient descent and track trajectory diagnostics.

    Tracks per-step:
        - cond_entropy: E_x[H(p(x))]
        - marginal_entropy: H(p_bar)
        - mutual_info: H(p_bar) - E_x[H(p(x))]
        - top_class_fraction: fraction of dominant predicted class
        - grad_norm: L2 norm of the gradient w.r.t adapted parameters ||grad||_2
        - accuracy: classification accuracy on y_eval (diagnostic only)
        - bias: parameter values b(t)
    """
    if adapt_mode == "bias_only":
        model.weight.requires_grad = False
        model.bias.requires_grad = True
        params = [model.bias]
    elif adapt_mode == "weights_and_bias":
        model.weight.requires_grad = True
        model.bias.requires_grad = True
        params = [model.weight, model.bias]
    else:
        raise ValueError(f"Unknown adapt_mode: {adapt_mode}. Choose 'bias_only' or 'weights_and_bias'.")

    optimizer = torch.optim.SGD(params, lr=learning_rate)

    history: Dict[str, List[Any]] = {
        "step": [],
        "cond_entropy": [],
        "marginal_entropy": [],
        "mutual_info": [],
        "top_class_fraction": [],
        "accuracy": [],
        "bias": [],
        "weight_norm": [],
        "grad_norm": [],
    }

    collapse_step: Optional[int] = None
    last_grad_norm: float = 0.0

    for step in range(num_steps):
        optimizer.zero_grad()
        logits = model(X)

        if objective == "tent":
            loss = compute_conditional_entropy(logits, reduction="mean")
            cond_ent = loss.item()
            marg_ent = compute_marginal_entropy(logits=logits).item()
        elif objective == "tent_im":
            loss, metrics = compute_im_loss(logits, lambda_im=lambda_im)
            cond_ent = metrics["cond_entropy"]
            marg_ent = metrics["marginal_entropy"]
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Backward pass to obtain exact gradient
        loss.backward()

        # Compute L2 norm of parameter gradient
        with torch.no_grad():
            grad_sq_sum = 0.0
            for p in params:
                if p.grad is not None:
                    grad_sq_sum += float(torch.sum(p.grad ** 2).item())
            current_grad_norm = float(np.sqrt(grad_sq_sum))
            last_grad_norm = current_grad_norm

            preds = torch.argmax(logits, dim=-1)
            acc = (preds == y_eval).float().mean().item()
            top_frac, _ = compute_top_class_fraction(logits)
            w_norm = torch.norm(model.weight).item()
            b_vec = model.bias.clone().detach().cpu().numpy()
            mi = marg_ent - cond_ent

            is_now_collapsed = check_collapse_criteria(
                top_frac, marg_ent, model.num_classes, definition=collapse_definition
            )
            if collapse_step is None and is_now_collapsed:
                collapse_step = step

        if (step % log_every == 0) or (step == num_steps - 1):
            history["step"].append(step)
            history["cond_entropy"].append(cond_ent)
            history["marginal_entropy"].append(marg_ent)
            history["mutual_info"].append(mi)
            history["top_class_fraction"].append(top_frac)
            history["accuracy"].append(acc)
            history["bias"].append(b_vec)
            history["weight_norm"].append(w_norm)
            history["grad_norm"].append(current_grad_norm)

        # Gradient update step
        optimizer.step()

    final_top_frac = history["top_class_fraction"][-1]
    final_marg_ent = history["marginal_entropy"][-1]

    # Evaluate all collapse definitions at final step for multi-definition analysis
    collapsed_dict = {
        "standard": check_collapse_criteria(final_top_frac, final_marg_ent, model.num_classes, "standard"),
        "def_a": check_collapse_criteria(final_top_frac, final_marg_ent, model.num_classes, "def_a"),
        "def_b": check_collapse_criteria(final_top_frac, final_marg_ent, model.num_classes, "def_b"),
        "def_c": check_collapse_criteria(final_top_frac, final_marg_ent, model.num_classes, "def_c"),
    }

    return {
        "steps": np.array(history["step"]),
        "cond_entropy": np.array(history["cond_entropy"]),
        "marginal_entropy": np.array(history["marginal_entropy"]),
        "mutual_info": np.array(history["mutual_info"]),
        "top_class_fraction": np.array(history["top_class_fraction"]),
        "accuracy": np.array(history["accuracy"]),
        "bias_history": np.array(history["bias"]),
        "weight_norm_history": np.array(history["weight_norm"]),
        "grad_norm_history": np.array(history["grad_norm"]),
        "is_collapsed": bool(collapsed_dict[collapse_definition]),
        "collapsed_dict": collapsed_dict,
        "collapse_step": collapse_step,
        "final_accuracy": float(history["accuracy"][-1]),
        "final_top_fraction": float(final_top_frac),
        "final_marginal_entropy": float(final_marg_ent),
        "final_grad_norm": float(last_grad_norm),
    }


def simulate_bias_dynamics_fast(
    precomputed_F: torch.Tensor,  # (N, K) = X @ W.T
    initial_bias: torch.Tensor,   # (K,)
    y_eval: torch.Tensor,         # (N,)
    learning_rate: float = 0.05,
    num_steps: int = 10000,
    collapse_definition: str = "standard",
    log_every: int = 50,
    early_stop_on_collapse: bool = False,
) -> Dict[str, Any]:
    """High-speed vectorized tensor simulator for bias-only entropy minimisation.

    Precomputes F = X @ W.T once.
    At each step, directly applies the analytical gradient:
        grad_b = - (1/N) * sum_i [ p(x_i) * (ln p(x_i) + H(p(x_i))) ]
    Runs up to 50,000 steps in a few seconds on CPU/GPU.
    If early_stop_on_collapse is True, immediately returns once collapse criteria is satisfied.
    """
    device = precomputed_F.device
    num_classes = precomputed_F.size(-1)
    b = initial_bias.clone().to(device)
    max_entropy = float(np.log(num_classes))

    history_steps = []
    history_h_pbar = []
    history_cond_ent = []
    history_top_frac = []
    history_grad_norm = []
    history_bias = []

    collapse_step: Optional[int] = None
    last_grad_norm: float = 0.0
    is_collapsed_flag: bool = False

    for step in range(num_steps):
        logits = precomputed_F + b
        log_P = F.log_softmax(logits, dim=-1)
        P = torch.exp(log_P)

        # H per sample: (N, 1)
        plogp = torch.where(P > 0.0, P * log_P, torch.zeros_like(P))
        H = -torch.sum(plogp, dim=-1, keepdim=True)

        # Analytical gradient w.r.t b: (K,)
        grad = -torch.mean(P * (log_P + H), dim=0)
        grad_norm = float(torch.norm(grad).item())
        last_grad_norm = grad_norm

        if (step % log_every == 0) or (step == num_steps - 1):
            p_bar = torch.mean(P, dim=0)
            h_pbar = -float(torch.sum(torch.where(p_bar > 0.0, p_bar * torch.log(p_bar), torch.zeros_like(p_bar))).item())
            preds = torch.argmax(logits, dim=-1)
            counts = torch.bincount(preds, minlength=num_classes)
            top_frac = float(torch.max(counts).item()) / float(preds.size(0))

            history_steps.append(step)
            history_h_pbar.append(h_pbar)
            history_cond_ent.append(float(H.mean().item()))
            history_top_frac.append(top_frac)
            history_grad_norm.append(grad_norm)
            history_bias.append(b.clone().detach().cpu().numpy())

            is_now_collapsed = check_collapse_criteria(top_frac, h_pbar, num_classes, definition=collapse_definition)
            if collapse_step is None and is_now_collapsed:
                collapse_step = step
                is_collapsed_flag = True

            if early_stop_on_collapse and is_now_collapsed:
                break

        # Gradient update
        b = b - learning_rate * grad

    # Final evaluation
    logits = precomputed_F + b
    log_P = F.log_softmax(logits, dim=-1)
    P = torch.exp(log_P)
    p_bar = torch.mean(P, dim=0)
    final_h_pbar = -float(torch.sum(torch.where(p_bar > 0.0, p_bar * torch.log(p_bar), torch.zeros_like(p_bar))).item())
    preds = torch.argmax(logits, dim=-1)
    counts = torch.bincount(preds, minlength=num_classes)
    final_top_frac = float(torch.max(counts).item()) / float(preds.size(0))
    final_acc = float((preds == y_eval).float().mean().item())

    collapsed_dict = {
        "standard": check_collapse_criteria(final_top_frac, final_h_pbar, num_classes, "standard"),
        "def_a": check_collapse_criteria(final_top_frac, final_h_pbar, num_classes, "def_a"),
        "def_b": check_collapse_criteria(final_top_frac, final_h_pbar, num_classes, "def_b"),
        "def_c": check_collapse_criteria(final_top_frac, final_h_pbar, num_classes, "def_c"),
    }

    return {
        "steps": np.array(history_steps),
        "cond_entropy": np.array(history_cond_ent),
        "marginal_entropy": np.array(history_h_pbar),
        "top_class_fraction": np.array(history_top_frac),
        "grad_norm_history": np.array(history_grad_norm),
        "bias_history": np.array(history_bias),
        "is_collapsed": bool(collapsed_dict[collapse_definition]),
        "collapsed_dict": collapsed_dict,
        "collapse_step": collapse_step,
        "final_accuracy": final_acc,
        "final_top_fraction": final_top_frac,
        "final_marginal_entropy": final_h_pbar,
        "final_grad_norm": last_grad_norm,
    }
