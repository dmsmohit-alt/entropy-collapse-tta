"""Basin scan and aggregation for the toy entropy-minimisation dynamics."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from theory.toy_model import ToySetup, make_gaussian_mixture, simulate_entropy_minimisation, source_metrics


def make_bias_grid(start: float, stop: float, step: float) -> tuple[float, ...]:
    """Build an inclusive, rounded bias grid and reject malformed ranges."""
    values = (start, stop, step)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bias grid bounds and step must be finite")
    if step <= 0:
        raise ValueError("bias grid step must be > 0")
    if stop < start:
        raise ValueError("bias grid stop must be >= start")
    count = int(round((stop - start) / step)) + 1
    grid = tuple(round(start + index * step, 10) for index in range(count))
    if not grid or grid[-1] < stop - (step / 2):
        raise ValueError("bias grid did not reach the requested stop")
    return grid


def validate_seeds(seeds: Iterable[Any]) -> tuple[int, ...]:
    """Validate a non-empty list of unique integer seeds."""
    validated: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("every seed must be an integer")
        if not 0 <= seed < 2**32:
            raise ValueError("every seed must be in [0, 2**32)")
        validated.append(seed)
    if not validated:
        raise ValueError("at least one seed is required")
    if len(set(validated)) != len(validated):
        raise ValueError("seeds must be unique")
    return tuple(validated)


def run_basin_scan(
    setup: ToySetup,
    *,
    seeds: Sequence[int],
    biases: Sequence[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every ``(seed, bias)`` pair and return scan rows plus source rows."""
    scan_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for seed in seeds:
        data = make_gaussian_mixture(setup, seed=seed)
        source = source_metrics(data, setup)
        source_rows.append({"seed": seed, **source})

        for bias in biases:
            result = simulate_entropy_minimisation(data, setup, initial_bias=bias)
            final = result.final
            scan_rows.append(
                {
                    "seed": seed,
                    "initial_bias": float(bias),
                    "collapsed": int(result.collapse_step is not None),
                    "collapse_step": result.collapse_step,
                    "collapse_onset_step": result.onset_step,
                    "final_marginal_entropy": float(final["eval_marginal_entropy"]),
                    "final_top_class_fraction": float(final["eval_top_class_fraction"]),
                    "final_accuracy": float(final["eval_accuracy"]),
                    "final_bias_max_minus_mean": float(final["bias_max_minus_mean"]),
                    "final_weight_frobenius": float(final["weight_frobenius"]),
                }
            )
    return scan_rows, source_rows


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return mean, std


def _metric_summary(values: Sequence[float]) -> dict[str, float]:
    mean, std = _mean_std(values)
    return {"mean": mean, "std": std}


def summarize_basin(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_rows: Sequence[Mapping[str, Any]],
    setup: ToySetup,
    seeds: Sequence[int],
    biases: Sequence[float],
) -> dict[str, Any]:
    """Aggregate per-run scan rows across seeds.

    The reported 50% crossing is descriptive.  It is defined as the midpoint
    between the largest tested bias with collapse rate below 0.5 and the
    smallest tested bias with collapse rate at least 0.5.  It is not a
    derived threshold and it is not interpreted as exact.
    """
    grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["initial_bias"])].append(row)

    aggregate: list[dict[str, Any]] = []
    for bias in biases:
        subset = grouped.get(float(bias), [])
        if not subset:
            continue
        collapsed = [int(row["collapsed"]) for row in subset]
        collapse_steps = [
            int(row["collapse_step"]) for row in subset if row["collapse_step"] is not None
        ]
        aggregate.append(
            {
                "initial_bias": float(bias),
                "n_seeds": len(subset),
                "collapse_rate": float(np.mean(collapsed)),
                "collapsed_by_seed": collapsed,
                "final_marginal_entropy": _metric_summary(
                    [float(row["final_marginal_entropy"]) for row in subset]
                ),
                "final_top_class_fraction": _metric_summary(
                    [float(row["final_top_class_fraction"]) for row in subset]
                ),
                "final_accuracy": _metric_summary(
                    [float(row["final_accuracy"]) for row in subset]
                ),
                "collapse_step": _metric_summary(collapse_steps) if collapse_steps else None,
            }
        )

    rates = [float(entry["collapse_rate"]) for entry in aggregate]
    crossing_index = next((index for index, rate in enumerate(rates) if rate >= 0.5), None)
    if crossing_index is None:
        threshold = None
    else:
        if crossing_index == 0:
            lower = upper = float(aggregate[0]["initial_bias"])
        else:
            lower = float(aggregate[crossing_index - 1]["initial_bias"])
            upper = float(aggregate[crossing_index]["initial_bias"])
        threshold = {
            "definition": (
                "midpoint between largest tested bias with collapse rate < 0.5 "
                "and smallest tested bias with collapse rate >= 0.5"
            ),
            "midpoint": (lower + upper) / 2.0,
            "bracketing_interval": [lower, upper],
            "half_width": (upper - lower) / 2.0,
        }

    no_collapse = [
        float(entry["initial_bias"]) for entry in aggregate if entry["collapse_rate"] == 0.0
    ]
    full_collapse = [
        float(entry["initial_bias"]) for entry in aggregate if entry["collapse_rate"] == 1.0
    ]

    source_metrics_summary = {}
    for key in (
        "mean_predictive_entropy",
        "marginal_entropy",
        "entropy_gap",
        "top_class_fraction",
        "accuracy",
        "mean_max_probability",
    ):
        source_metrics_summary[key] = _metric_summary([float(row[key]) for row in source_rows])

    return {
        "seeds": list(seeds),
        "initial_bias_grid": {
            "min": float(biases[0]),
            "max": float(biases[-1]),
            "step": float(biases[1] - biases[0]) if len(biases) > 1 else 0.0,
            "count": len(biases),
        },
        "collapse_criteria": asdict(setup.collapse),
        "threshold": threshold,
        "no_collapse_up_to": max(no_collapse) if no_collapse else None,
        "full_collapse_from": min(full_collapse) if full_collapse else None,
        "source": source_metrics_summary,
        "basin_summary": aggregate,
    }


def write_scan_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write per-run rows to CSV with a stable column order."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "initial_bias",
        "collapsed",
        "collapse_step",
        "collapse_onset_step",
        "final_marginal_entropy",
        "final_top_class_fraction",
        "final_accuracy",
        "final_bias_max_minus_mean",
        "final_weight_frobenius",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return destination


def write_summary_csv(summary: Mapping[str, Any], path: str | Path) -> Path:
    """Write the bias-level aggregate table to CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "initial_bias",
        "n_seeds",
        "collapse_rate",
        "final_marginal_entropy_mean",
        "final_marginal_entropy_std",
        "final_top_class_fraction_mean",
        "final_top_class_fraction_std",
        "final_accuracy_mean",
        "final_accuracy_std",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["basin_summary"]:
            writer.writerow(
                {
                    "initial_bias": row["initial_bias"],
                    "n_seeds": row["n_seeds"],
                    "collapse_rate": row["collapse_rate"],
                    "final_marginal_entropy_mean": row["final_marginal_entropy"]["mean"],
                    "final_marginal_entropy_std": row["final_marginal_entropy"]["std"],
                    "final_top_class_fraction_mean": row["final_top_class_fraction"]["mean"],
                    "final_top_class_fraction_std": row["final_top_class_fraction"]["std"],
                    "final_accuracy_mean": row["final_accuracy"]["mean"],
                    "final_accuracy_std": row["final_accuracy"]["std"],
                }
            )
    return destination


def write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a JSON artifact without silently allowing NaN values."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return destination


__all__ = [
    "make_bias_grid",
    "run_basin_scan",
    "summarize_basin",
    "validate_seeds",
    "write_json",
    "write_scan_csv",
    "write_summary_csv",
]
