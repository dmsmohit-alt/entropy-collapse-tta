"""Tests for the shared collapse detector."""

from __future__ import annotations

import pytest

from metrics.collapse import CollapseCriteria, CollapseDetector


def test_collapse_requires_persistence_and_reports_steps() -> None:
    criteria = CollapseCriteria(
        marginal_entropy_threshold=0.10,
        top_class_fraction_threshold=0.95,
        persistence=3,
    )
    detector = CollapseDetector(criteria)

    assert detector.update(0, 0.05, 0.96) is False
    assert detector.update(10, 0.04, 0.97) is False
    assert detector.consecutive == 2
    assert detector.update(20, 0.03, 0.98) is True

    assert detector.collapsed_at == 20
    assert detector.onset_step == 0
    assert detector.is_collapsed
    assert detector.update(30, 0.01, 0.99) is False


def test_non_collapse_evaluation_resets_the_window() -> None:
    detector = CollapseDetector(
        CollapseCriteria(
            marginal_entropy_threshold=0.10,
            top_class_fraction_threshold=0.95,
            persistence=3,
        )
    )

    detector.update(0, 0.05, 0.96)
    detector.update(10, 0.05, 0.96)
    assert detector.consecutive == 2
    detector.update(20, 0.30, 0.96)
    assert detector.consecutive == 0
    assert detector.onset_step is None

    detector.update(30, 0.05, 0.96)
    detector.update(40, 0.05, 0.96)
    assert not detector.is_collapsed
    assert detector.update(50, 0.05, 0.96) is True
    assert detector.onset_step == 30


def test_high_top_fraction_without_entropy_collapse_is_not_enough() -> None:
    detector = CollapseDetector(CollapseCriteria(persistence=2))
    assert detector.update(0, 0.50, 0.99) is False
    assert detector.update(10, 0.50, 0.99) is False
    assert not detector.is_collapsed


def test_invalid_detector_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        CollapseCriteria(marginal_entropy_threshold=-1)
    with pytest.raises(ValueError):
        CollapseCriteria(top_class_fraction_threshold=1.1)
    with pytest.raises(ValueError):
        CollapseCriteria(persistence=0)

    detector = CollapseDetector()
    with pytest.raises(ValueError):
        detector.update(0, -0.1, 0.99)
    with pytest.raises(ValueError):
        detector.update(0, 0.05, 1.1)
    detector.update(1, 0.05, 0.99)
    with pytest.raises(ValueError, match="strictly increasing"):
        detector.update(1, 0.05, 0.99)
