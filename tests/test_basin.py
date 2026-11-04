"""Tests for bias-grid construction and basin aggregation."""

from __future__ import annotations

import pytest

from theory.basin import (
    make_bias_grid,
    summarize_basin,
    validate_seeds,
    write_scan_csv,
    write_summary_csv,
)
from theory.toy_model import ToySetup


def test_bias_grid_is_inclusive_and_rounded() -> None:
    grid = make_bias_grid(0.0, 0.3, 0.1)
    assert grid == pytest.approx((0.0, 0.1, 0.2, 0.3))

    with pytest.raises(ValueError, match="step"):
        make_bias_grid(0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="stop"):
        make_bias_grid(1.0, 0.0, 0.1)


def test_validate_seeds() -> None:
    assert validate_seeds([0, 1, 2]) == (0, 1, 2)
    with pytest.raises(ValueError):
        validate_seeds([])
    with pytest.raises(ValueError):
        validate_seeds([0, 0])
    with pytest.raises(ValueError):
        validate_seeds([True])


def _row(seed: int, bias: float, collapsed: int) -> dict[str, object]:
    return {
        "seed": seed,
        "initial_bias": bias,
        "collapsed": collapsed,
        "collapse_step": 500 if collapsed else None,
        "collapse_onset_step": 300 if collapsed else None,
        "final_marginal_entropy": 0.01 if collapsed else 1.09,
        "final_top_class_fraction": 1.0 if collapsed else 0.34,
        "final_accuracy": 1 / 3 if collapsed else 0.97,
        "final_bias_max_minus_mean": 5.0 if collapsed else 1.0,
        "final_weight_frobenius": 2.0 if collapsed else 8.0,
    }


def _source(seed: int) -> dict[str, object]:
    return {
        "seed": seed,
        "mean_predictive_entropy": 0.2,
        "marginal_entropy": 1.098,
        "entropy_gap": 0.9,
        "top_class_fraction": 0.34,
        "accuracy": 0.97,
        "mean_max_probability": 0.96,
    }


def test_summary_finds_50_percent_crossing() -> None:
    biases = (0.0, 1.0, 2.0, 3.0)
    rows = []
    for seed in (0, 1, 2):
        rows.extend(
            [
                _row(seed, 0.0, 0),
                _row(seed, 1.0, 0),
                _row(seed, 2.0, int(seed == 0)),
                _row(seed, 3.0, 1),
            ]
        )

    summary = summarize_basin(
        rows,
        source_rows=[_source(seed) for seed in (0, 1, 2)],
        setup=ToySetup(),
        seeds=(0, 1, 2),
        biases=biases,
    )

    assert [row["collapse_rate"] for row in summary["basin_summary"]] == pytest.approx(
        [0.0, 0.0, 1 / 3, 1.0]
    )
    assert summary["threshold"]["midpoint"] == pytest.approx(2.5)
    assert summary["threshold"]["bracketing_interval"] == [2.0, 3.0]
    assert summary["no_collapse_up_to"] == 1.0
    assert summary["full_collapse_from"] == 3.0
    assert summary["source"]["accuracy"]["mean"] == pytest.approx(0.97)


def test_basin_csvs_round_trip_with_expected_columns(tmp_path) -> None:
    biases = (0.0, 1.0)
    rows = [_row(0, 0.0, 0), _row(1, 0.0, 0), _row(0, 1.0, 1), _row(1, 1.0, 1)]
    summary = summarize_basin(
        rows,
        source_rows=[_source(0), _source(1)],
        setup=ToySetup(),
        seeds=(0, 1),
        biases=biases,
    )

    scan_path = write_scan_csv(rows, tmp_path / "scan.csv")
    summary_path = write_summary_csv(summary, tmp_path / "summary.csv")

    assert scan_path.read_text(encoding="utf-8").splitlines()[0].startswith("seed,initial_bias")
    assert summary_path.read_text(encoding="utf-8").splitlines()[0].startswith("initial_bias")
    assert len(summary_path.read_text(encoding="utf-8").splitlines()) == 3
