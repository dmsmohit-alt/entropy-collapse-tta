"""Unit tests for information-theoretic metrics and collapse detector."""

import math
import pytest
import torch
import torch.nn.functional as F

from src.metrics.entropy import (
    compute_conditional_entropy,
    compute_marginal_distribution,
    compute_marginal_entropy,
    compute_im_loss,
    compute_top_class_fraction,
)
from src.metrics.collapse import CollapseDetector, EMAMarginalTracker


class TestEntropyMetrics:
    """Test suite for entropy and information metrics."""

    @pytest.mark.parametrize("num_classes", [2, 5, 10, 100])
    def test_uniform_entropy_equals_log_k(self, num_classes: int) -> None:
        """Verify that uniform logits yield H(p) = ln(K)."""
        batch_size = 8
        # Identical logits -> uniform distribution
        logits = torch.zeros(batch_size, num_classes)
        expected_entropy = math.log(num_classes)

        cond_ent = compute_conditional_entropy(logits, reduction="mean")
        assert torch.isclose(cond_ent, torch.tensor(expected_entropy, dtype=torch.float32), atol=1e-5)

        marg_ent = compute_marginal_entropy(logits=logits)
        assert torch.isclose(marg_ent, torch.tensor(expected_entropy, dtype=torch.float32), atol=1e-5)

    def test_deterministic_distribution_has_zero_entropy(self) -> None:
        """Verify that near-one-hot logits have H(p) close to 0."""
        # Class 0 strongly dominant
        logits = torch.tensor([[50.0, -50.0, -50.0], [ -50.0, 50.0, -50.0]])
        cond_ent = compute_conditional_entropy(logits, reduction="mean")
        assert cond_ent.item() < 1e-4

    def test_entropy_bounds(self) -> None:
        """Verify 0 <= H(p) <= ln(K) for arbitrary logits."""
        torch.manual_seed(42)
        logits = torch.randn(64, 10) * 5.0
        max_possible = math.log(10)

        ent_per_sample = compute_conditional_entropy(logits, reduction="none")
        assert (ent_per_sample >= -1e-6).all(), "Entropy cannot be negative"
        assert (ent_per_sample <= max_possible + 1e-5).all(), "Entropy cannot exceed ln(K)"

    def test_extreme_logits_no_nans(self) -> None:
        """Ensure numerical stability under extreme logit ranges (+-1000)."""
        logits = torch.tensor([
            [1000.0, -1000.0, -1000.0],
            [-1000.0, -1000.0, -1000.0],  # uniform but all extreme negative
            [0.0, 500.0, -500.0],
        ])
        ent = compute_conditional_entropy(logits, reduction="none")
        assert not torch.isnan(ent).any(), "NaN found in conditional entropy under extreme logits"
        assert not torch.isinf(ent).any(), "Inf found in conditional entropy under extreme logits"

        marg_ent = compute_marginal_entropy(logits=logits)
        assert not torch.isnan(marg_ent), "NaN found in marginal entropy under extreme logits"

    def test_marginal_distribution_shape_and_sum(self) -> None:
        """Check p_bar sums to 1.0 and has shape (K,)."""
        logits = torch.randn(32, 7)
        p_bar = compute_marginal_distribution(logits)
        assert p_bar.shape == (7,)
        assert torch.isclose(p_bar.sum(), torch.tensor(1.0), atol=1e-5)
        assert (p_bar >= 0.0).all()

    def test_im_loss_identity_and_components(self) -> None:
        """Verify IM loss components: L_IM = H(p) - lambda * H(p_bar)."""
        logits = torch.randn(16, 4)
        loss, metrics = compute_im_loss(logits, lambda_im=1.0)

        expected_loss = metrics["cond_entropy"] - 1.0 * metrics["marginal_entropy"]
        assert math.isclose(loss.item(), expected_loss, abs_tol=1e-5)
        # Shannon mutual information I(X; Y) = H(p_bar) - E[H(p)] >= 0
        assert metrics["mutual_info"] >= -1e-5, "Mutual information must be non-negative"

    def test_autograd_matches_analytical_gradient(self) -> None:
        """Verify autograd of conditional entropy matches analytical dH/dz_i = -p_i (ln p_i + H)."""
        torch.manual_seed(123)
        z = torch.randn(1, 5, requires_grad=True, dtype=torch.float64)

        # Autograd gradient
        H = compute_conditional_entropy(z, reduction="sum")
        H.backward()
        autograd_grad = z.grad.clone()

        # Analytical gradient
        with torch.no_grad():
            p = F.softmax(z, dim=-1)
            ln_p = F.log_softmax(z, dim=-1)
            H_analytical = -(p * ln_p).sum(dim=-1, keepdim=True)
            analytical_grad = -p * (ln_p + H_analytical)

        assert torch.allclose(autograd_grad, analytical_grad, atol=1e-6)

    def test_top_class_fraction(self) -> None:
        """Verify fraction of predictions in top class."""
        # 3 predictions for class 1, 1 for class 0
        logits = torch.tensor([
            [1.0, 5.0],
            [2.0, 6.0],
            [0.5, 4.0],
            [10.0, 2.0],
        ])
        frac, dominant_class = compute_top_class_fraction(logits)
        assert dominant_class == 1
        assert math.isclose(frac, 0.75)


class TestCollapseDetector:
    """Test suite for collapse detection logic."""

    def test_collapse_triggered_after_patience(self) -> None:
        detector = CollapseDetector(threshold=0.80, patience=3)

        # Batch with 100% predictions for class 0
        collapsed_logits = torch.tensor([[10.0, 0.0]] * 10)
        # Batch balanced
        balanced_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]] * 5)

        # Step 0: high bias, consecutive = 1 -> not collapsed yet
        c, step = detector.update(collapsed_logits, step=0)
        assert not c and step is None

        # Step 1: consecutive = 2 -> not collapsed yet
        c, step = detector.update(collapsed_logits, step=1)
        assert not c and step is None

        # Step 2: consecutive = 3 -> collapsed! First streak was at step 0
        c, step = detector.update(collapsed_logits, step=2)
        assert c is True
        assert step == 0

    def test_streak_broken_by_balanced_batch(self) -> None:
        detector = CollapseDetector(threshold=0.80, patience=3)
        collapsed_logits = torch.tensor([[10.0, 0.0]] * 10)
        balanced_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]] * 5)

        detector.update(collapsed_logits, step=0)
        detector.update(collapsed_logits, step=1)
        # Streak broken
        c, step = detector.update(balanced_logits, step=2)
        assert not c and step is None
        assert detector.consecutive_count == 0


class TestEMAMarginalTracker:
    """Test suite for EMA marginal distribution tracking."""

    def test_ema_tracking_convergence(self) -> None:
        tracker = EMAMarginalTracker(num_classes=2, alpha=0.5)

        # Send 5 batches with logits pointing strictly to class 0
        logits_class_0 = torch.tensor([[10.0, -10.0]] * 8)
        for _ in range(10):
            ema = tracker.update(logits_class_0)

        # Marginal should be very close to [1.0, 0.0]
        assert ema[0].item() > 0.99
        # Entropy should be near 0
        assert tracker.get_entropy() < 0.05
