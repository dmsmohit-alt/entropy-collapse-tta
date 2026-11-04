"""Linear-softmax model for theoretical dynamical analysis.

Implements:
    p(y|x) = softmax(W x + b)
with source-training initialization and configurable initial bias offsets.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSoftmaxModel(nn.Module):
    """Linear classifier with softmax probability outputs.

    z = x @ W.T + b
    p(y=k|x) = softmax(z)_k

    Attributes:
        num_classes: Number of classes K.
        dim: Input feature dimension d.
        weight: Parameter tensor of shape (K, d).
        bias: Parameter tensor of shape (K,).
    """

    def __init__(self, num_classes: int, dim: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.weight = nn.Parameter(torch.zeros(num_classes, dim))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute unnormalised logits z = x @ W.T + b."""
        return F.linear(x, self.weight, self.bias)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class probabilities p(y|x) = softmax(z)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def fit_source(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lr: float = 0.1,
        steps: int = 200,
        weight_decay: float = 1e-4,
    ) -> float:
        """Pretrain model on source data via cross-entropy to achieve high initial accuracy.

        Args:
            X: Source inputs (N, d).
            y: Source labels (N,).
            lr: Learning rate.
            steps: Optimization steps.
            weight_decay: L2 regularization coefficient.

        Returns:
            Final source classification accuracy [0.0, 1.0].
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        for _ in range(steps):
            optimizer.zero_grad()
            logits = self.forward(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(self.forward(X), dim=-1)
            acc = (preds == y).float().mean().item()

        return float(acc)

    def fit_centroid_solution(self, centroids: torch.Tensor, scale: float = 1.0) -> None:
        """Initialize weights analytically directly from Gaussian centroids.

        For spherical Gaussians with equal priors, the Bayes optimal decision
        boundary has linear weights W_k proportional to centroid mu_k.
        """
        with torch.no_grad():
            self.weight.copy_(centroids * scale)
            self.bias.zero_()

    def apply_bias_offset(self, delta: float, class_idx: int = 0) -> None:
        """Apply an initial bias perturbation: b[class_idx] += delta."""
        with torch.no_grad():
            self.bias[class_idx] += delta

    def clone(self) -> "LinearSoftmaxModel":
        """Return a deep copy of the model with detached parameters."""
        new_model = LinearSoftmaxModel(self.num_classes, self.dim)
        with torch.no_grad():
            new_model.weight.copy_(self.weight)
            new_model.bias.copy_(self.bias)
        return new_model
