"""Generate Phase 1 figures from saved run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory produced by run_theory.py")
    parser.add_argument(
        "--trajectory-seed",
        type=int,
        default=0,
        help="Seed to highlight in the trajectory figure.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _plot_basin(summary: dict[str, Any], output: Path) -> None:
    rows = summary["basin_summary"]
    biases = np.asarray([row["initial_bias"] for row in rows])
    entropy_mean = np.asarray([row["final_marginal_entropy"]["mean"] for row in rows])
    entropy_std = np.asarray([row["final_marginal_entropy"]["std"] for row in rows])
    top_mean = np.asarray([row["final_top_class_fraction"]["mean"] for row in rows])
    top_std = np.asarray([row["final_top_class_fraction"]["std"] for row in rows])
    collapse_rate = np.asarray([row["collapse_rate"] for row in rows])

    threshold = summary.get("threshold")
    crossing = float(threshold["midpoint"]) if threshold else None
    bracket = threshold["bracketing_interval"] if threshold else None

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 3.9), constrained_layout=True)

    axes[0].plot(biases, entropy_mean, color="#1f77b4", linewidth=1.8, label="mean across seeds")
    axes[0].fill_between(
        biases,
        entropy_mean - entropy_std,
        entropy_mean + entropy_std,
        color="#1f77b4",
        alpha=0.18,
        linewidth=0,
    )
    axes[0].set_ylabel(r"final $H(\bar p)$")
    axes[0].set_title("Marginal entropy after adaptation")

    axes[1].plot(biases, top_mean, color="#d62728", linewidth=1.8, label="mean across seeds")
    axes[1].fill_between(
        biases,
        top_mean - top_std,
        top_mean + top_std,
        color="#d62728",
        alpha=0.18,
        linewidth=0,
    )
    axes[1].axhline(0.95, color="0.35", linestyle=":", linewidth=1.0, label="0.95 criterion")
    axes[1].set_ylim(0.25, 1.02)
    axes[1].set_ylabel("final top-class fraction")
    axes[1].set_title("Prediction concentration")

    axes[2].step(biases, collapse_rate, where="mid", color="#2ca02c", linewidth=1.8)
    axes[2].scatter(biases, collapse_rate, s=14, color="#2ca02c")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("collapse rate")
    axes[2].set_title("Collapse frequency across seeds")

    for axis in axes:
        axis.set_xlabel("initial bias towards class 0")
        axis.grid(alpha=0.25, linewidth=0.6)
        if crossing is not None:
            axis.axvline(crossing, color="0.2", linestyle="--", linewidth=1.2)
        if bracket is not None and bracket[0] != bracket[1]:
            axis.axvspan(bracket[0], bracket[1], color="0.2", alpha=0.08)

    if crossing is not None:
        axes[2].annotate(
            f"50% crossing\n{crossing:.3f}",
            xy=(crossing, 0.5),
            xytext=(8, 22),
            textcoords="offset points",
            fontsize=8.5,
            arrowprops={"arrowstyle": "-", "color": "0.3", "linewidth": 0.8},
        )

    figure.suptitle(
        "Toy model: entropy-minimisation collapse basin\n"
        f"K={summary['toy_setup']['num_classes']}, d={summary['toy_setup']['dim']}, "
        f"separation={summary['toy_setup']['separation']}, "
        f"lr={summary['toy_setup']['learning_rate']}, steps={summary['toy_setup']['steps']}",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_trajectories(summary: dict[str, Any], run_dir: Path, seed: int, output: Path) -> None:
    trajectory_rows = [row for row in summary["trajectories"] if int(row["seed"]) == seed]
    if not trajectory_rows:
        raise ValueError(f"No trajectory rows for seed {seed}")

    biases = sorted({float(row["initial_bias"]) for row in trajectory_rows})
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(biases)))

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), constrained_layout=True)
    for bias, color in zip(biases, colors):
        row = next(row for row in trajectory_rows if float(row["initial_bias"]) == bias)
        records = _read_jsonl(run_dir / row["path"])
        steps = [record["step"] for record in records]
        entropy = [record["eval_marginal_entropy"] for record in records]
        top = [record["eval_top_class_fraction"] for record in records]
        collapse_step = row["collapse_step"]
        label = f"bias={bias:.2f}" + (f", collapse@{collapse_step}" if collapse_step else ", no collapse")
        axes[0].plot(steps, entropy, color=color, linewidth=1.7, label=label)
        axes[1].plot(steps, top, color=color, linewidth=1.7, label=label)
        if collapse_step is not None:
            axes[0].axvline(collapse_step, color=color, linestyle=":", linewidth=0.9, alpha=0.7)
            axes[1].axvline(collapse_step, color=color, linestyle=":", linewidth=0.9, alpha=0.7)

    axes[0].set_ylabel(r"$H(\bar p)$ on held-out data")
    axes[0].set_title("Marginal entropy trajectory")
    axes[1].set_ylabel("top-class fraction")
    axes[1].set_title("Prediction concentration trajectory")
    axes[1].set_ylim(0.25, 1.02)
    axes[1].axhline(0.95, color="0.35", linestyle=":", linewidth=1.0)

    for axis in axes:
        axis.set_xlabel("adaptation step")
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.legend(fontsize=8, loc="best")

    figure.suptitle(f"Toy entropy-minimisation trajectories (data seed {seed})", fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    figure_dir = run_dir / "figures"
    basin_path = figure_dir / "basin_boundary.png"
    trajectory_path = figure_dir / "trajectories.png"
    _plot_basin(summary, basin_path)
    _plot_trajectories(summary, run_dir, args.trajectory_seed, trajectory_path)

    print(f"Basin figure:      {basin_path.resolve()}")
    print(f"Trajectory figure: {trajectory_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
