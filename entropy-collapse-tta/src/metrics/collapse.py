"""Collapse detection and tracking metrics for continual test-time adaptation.

Defines:
- CollapseDetector: tracks consecutive batches exceeding dominant class fraction threshold
  to define a rigorous "time-to-collapse" metric.
- EMAMarginalTracker: maintains an exponential moving average of the marginal prediction
  distribution p_bar across streaming batches.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


class CollapseDetector:
    """Detects adaptation collapse based on sustained single-class dominance.

    Definition:
        A model has 'collapsed' at step t if the fraction of predictions
        assigned to the top class exceeds `threshold` (e.g. 0.90) for
        `patience` consecutive batches.

    Attributes:
        threshold: Fraction cutoff (e.g. 0.90 means >=90% of batch is one class).
        patience: Required number of consecutive batches exceeding threshold.
        collapse_step: Step index of the first batch in the collapse sequence,
                       or None if no collapse occurred.
        collapsed: Boolean flag indicating if collapse criteria has been met.
    """

    def __init__(self, threshold: float = 0.90, patience: int = 5) -> None:
        self.threshold = threshold
        self.patience = patience
        self.consecutive_count: int = 0
        self.first_streak_step: Optional[int] = None
        self.collapse_step: Optional[int] = None
        self.collapsed: bool = False
        self.dominant_class_history: List[int] = []

    def update(self, logits: torch.Tensor, step: int) -> Tuple[bool, Optional[int]]:
        """Process a batch and check for collapse.

        Args:
            logits: (B, K) unnormalised model predictions.
            step: Current streaming batch step index.

        Returns:
            (is_currently_collapsed, collapse_step)
        """
        preds = torch.argmax(logits, dim=-1)
        batch_size = preds.size(0)
        num_classes = logits.size(-1)
        counts = torch.bincount(preds, minlength=num_classes)
        max_count, dominant_class = torch.max(counts, dim=0)
        fraction = float(max_count.item()) / float(batch_size)

        self.dominant_class_history.append(int(dominant_class.item()))

        if fraction >= self.threshold:
            if self.consecutive_count == 0:
                self.first_streak_step = step
            self.consecutive_count += 1
            if self.consecutive_count >= self.patience and not self.collapsed:
                self.collapsed = True
                self.collapse_step = self.first_streak_step
        else:
            self.consecutive_count = 0
            self.first_streak_step = None

        return self.collapsed, self.collapse_step

    def reset(self) -> None:
        """Reset internal detector state for a fresh run."""
        self.consecutive_count = 0
        self.first_streak_step = None
        self.collapse_step = None
        self.collapsed = False
        self.dominant_class_history.clear()


class EMAMarginalTracker:
    """Tracks the running marginal class distribution p_bar via exponential moving average.

    p_bar_{t} = (1 - alpha) * p_bar_{t-1} + alpha * p_bar_batch

    Useful on small batches where single-batch marginal entropy is noisy.
    """

    def __init__(self, num_classes: int, alpha: float = 0.1) -> None:
        self.num_classes = num_classes
        self.alpha = alpha
        self.ema_p_bar: Optional[torch.Tensor] = None

    def update(self, logits: torch.Tensor) -> torch.Tensor:
        """Update EMA with current batch logits.

        Args:
            logits: (B, K) batch predictions.

        Returns:
            ema_p_bar: (K,) updated running distribution.
        """
        probs = F.softmax(logits.detach(), dim=-1).mean(dim=0)
        if self.ema_p_bar is None:
            self.ema_p_bar = probs.clone()
        else:
            self.ema_p_bar = (1.0 - self.alpha) * self.ema_p_bar + self.alpha * probs
        return self.ema_p_bar

    def get_entropy(self, eps: float = 1e-12) -> float:
        """Compute entropy of the current EMA marginal distribution."""
        if self.ema_p_bar is None:
            return 0.0
        p = torch.clamp(self.ema_p_bar, min=eps)
        ent = -torch.sum(p * torch.log(p)).item()
        return float(ent)

    def reset(self) -> None:
        """Reset tracker state."""
        self.ema_p_bar = None
