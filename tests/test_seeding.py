"""Tests for deterministic seed handling."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from utils.seeding import seed_everything


def _draw_sequence() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.rand()), torch.rand(4)


def test_same_seed_reproduces_python_numpy_and_torch() -> None:
    seed_everything(123, deterministic=False)
    first = _draw_sequence()

    seed_everything(123, deterministic=False)
    second = _draw_sequence()

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_different_seed_changes_streams() -> None:
    seed_everything(123, deterministic=False)
    first = _draw_sequence()

    seed_everything(124, deterministic=False)
    second = _draw_sequence()

    assert first[0] != second[0]
    assert first[1] != second[1]
    assert not torch.equal(first[2], second[2])


def test_invalid_seed_is_rejected() -> None:
    with pytest.raises(TypeError):
        seed_everything(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        seed_everything(-1)
    with pytest.raises(ValueError):
        seed_everything(2**32)
