"""Generic, stateful collapse detection.

A single high-confidence batch is not called collapse.  Collapse is declared
only after both conditions hold for ``persistence`` consecutive evaluations:

    H(p_bar) <= marginal_entropy_threshold
    top_class_fraction >= top_class_fraction_threshold

The detector is independent of the toy model so the real-network stream can
use exactly the same definition in Phase 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CollapseCriteria:
    """Thresholds and persistence for a collapse declaration.

    ``persistence`` counts *consecutive calls* to ``CollapseDetector.update``.
    For a stream evaluated every ``eval_every`` steps, that corresponds to
    ``persistence * eval_every`` adaptation steps.
    """

    marginal_entropy_threshold: float = 0.10
    top_class_fraction_threshold: float = 0.95
    persistence: int = 20

    def __post_init__(self) -> None:
        if not math.isfinite(self.marginal_entropy_threshold) or self.marginal_entropy_threshold < 0:
            raise ValueError("marginal_entropy_threshold must be finite and >= 0")
        if not 0 < self.top_class_fraction_threshold <= 1:
            raise ValueError("top_class_fraction_threshold must be in (0, 1]")
        if isinstance(self.persistence, bool) or not isinstance(self.persistence, int):
            raise TypeError("persistence must be an integer")
        if self.persistence < 1:
            raise ValueError("persistence must be >= 1")


class CollapseDetector:
    """Track consecutive collapse-like evaluations.

    ``time_to_collapse`` is the evaluation step at which the condition has
    held for the required persistence.  ``onset_step`` is the first step of
    that accepted run.  Both are ``None`` until collapse is declared.
    """

    def __init__(self, criteria: CollapseCriteria | None = None) -> None:
        self.criteria = criteria or CollapseCriteria()
        self._consecutive = 0
        self._last_step: int | None = None
        self.collapsed_at: int | None = None
        self.onset_step: int | None = None

    @property
    def consecutive(self) -> int:
        return self._consecutive

    @property
    def is_collapsed(self) -> bool:
        return self.collapsed_at is not None

    def update(self, step: int, marginal_entropy: float, top_class_fraction: float) -> bool:
        """Update state and return True exactly when collapse is first declared."""
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer")
        if self._last_step is not None and step <= self._last_step:
            raise ValueError("step must be strictly increasing across updates")
        if not math.isfinite(marginal_entropy) or marginal_entropy < 0:
            raise ValueError("marginal_entropy must be finite and >= 0")
        if not math.isfinite(top_class_fraction) or not 0 <= top_class_fraction <= 1:
            raise ValueError("top_class_fraction must be finite and in [0, 1]")

        self._last_step = step
        if self.is_collapsed:
            return False

        meets = (
            marginal_entropy <= self.criteria.marginal_entropy_threshold
            and top_class_fraction >= self.criteria.top_class_fraction_threshold
        )
        if meets:
            if self._consecutive == 0:
                self.onset_step = step
            self._consecutive += 1
        else:
            self._consecutive = 0
            self.onset_step = None

        if self._consecutive >= self.criteria.persistence:
            self.collapsed_at = step
            return True
        return False


__all__ = ["CollapseCriteria", "CollapseDetector"]
