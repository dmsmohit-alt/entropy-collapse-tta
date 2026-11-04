"""Generate Phase 2 figures from saved stability artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory produced by run_stability.py",
    )
    return parser.parse_args()


def _read_hessian(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _plot_spectrum(rows: list[dict[str, str]], output: Path, num_classes: int) -> None:
    seed = int(rows[0]["seed"])
    rows = [row for row in rows if int(row["seed"]) == seed]
    lambdas = np.asarray([float(row["lambda"]) for row in rows])
    marginal = np.asarray([float(row["marginal_curvature"]) for row in rows])
    observed = np.asarray([float(row["b_eigenvalue_1"]) for row in rows])
    full_max = np.asarray([float(row["full_max_eigenvalue"]) for row in rows])
    predicted_full = np.asarray([float(row["predicted_full_max"]) for row in rows])

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)

    axes[0].axvspan(1.0, lambdas.max() + 0.2, color="#2ca02c", alpha=0.10)
    axes[0].axvspan(lambdas.min() - 0.2, 1.0, color="#d62728", alpha=0.08)
    axes[0].axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    axes[0].axvline(1.0, color="0.2", linestyle="--", linewidth=1.2)
    axes[0].plot(lambdas, marginal, color="#1f77b4", linewidth=1.8, label=r"derived $(\lambda-1)/K$")
    axes[0].scatter(lambdas, observed, color="#d62728", s=28, zorder=3, label="observed bias-block eigenvalue")
    axes[0].set_xlabel(r"information weight $\lambda$")
    axes[0].set_ylabel("marginal-mode curvature")
    axes[0].set_title("Bias-block Hessian at the symmetric point")
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].axvspan(1.0, lambdas.max() + 0.2, color="#2ca02c", alpha=0.10)
    axes[1].axvspan(lambdas.min() - 0.2, 1.0, color="#d62728", alpha=0.08)
    axes[1].axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    axes[1].axvline(1.0, color="0.2", linestyle="--", linewidth=1.2)
    axes[1].plot(
        lambdas,
        predicted_full,
        color="#1f77b4",
        linewidth=1.8,
        linestyle="--",
        label=r"derived $\max(0,(\lambda-1)/K)$",
    )
    axes[1].scatter(lambdas, full_max, color="#2ca02c", s=28, zorder=3, label="observed full Hessian")
    axes[1].set_xlabel(r"information weight $\lambda$")
    axes[1].set_ylabel("largest Hessian eigenvalue")
    axes[1].set_title("Centred features: only the marginal mode is positive")
    axes[1].legend(fontsize=8, loc="upper left")

    figure.suptitle(f"Phase 2 local stability checks (K={num_classes}, seed {seed})", fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_reduced(records: list[dict[str, Any]], output: Path, num_classes: int) -> None:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[float(record["lambda"])].append(record)

    lambdas = sorted(grouped)
    colors = plt.cm.viridis(np.linspace(0.10, 0.90, len(lambdas)))
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)

    for lam, color in zip(lambdas, colors):
        rows = sorted(grouped[lam], key=lambda item: item["step"])
        steps = np.asarray([row["step"] for row in rows])
        entropy = np.asarray([row["marginal_entropy"] for row in rows])
        spread = np.asarray([row["bias_spread"] for row in rows])
        label = rf"$\lambda={lam:.1f}$"
        axes[0].plot(steps, entropy, color=color, linewidth=1.7, label=label)
        axes[1].semilogy(steps, spread, color=color, linewidth=1.7, label=label)

    log_k = float(np.log(num_classes))
    axes[0].axhline(log_k, color="0.35", linestyle=":", linewidth=1.0, label=r"$\log K$")
    axes[0].set_xlabel("reduced-flow step")
    axes[0].set_ylabel(r"$H(p)$")
    axes[0].set_title("Reduced W=0 bias flow")
    axes[0].legend(fontsize=8, loc="best")

    axes[1].set_xlabel("reduced-flow step")
    axes[1].set_ylabel(r"$\|b-\bar b\|_2$")
    axes[1].set_title("Bias spread (log scale)")
    axes[1].legend(fontsize=8, loc="best")

    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.6)

    figure.suptitle("Phase 2 reduced bias dynamics: lambda = 1 separates collapse from balance", fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    num_classes = int(summary["stability_spec"]["num_classes"])

    spectrum_path = run_dir / "figures" / "stability_spectrum.png"
    reduced_path = run_dir / "figures" / "reduced_dynamics.png"

    _plot_spectrum(
        _read_hessian(run_dir / "hessian_spectrum.csv"),
        spectrum_path,
        num_classes,
    )
    _plot_reduced(
        _read_jsonl(run_dir / "reduced_trajectories.jsonl"),
        reduced_path,
        num_classes,
    )

    print(f"Spectrum figure: {spectrum_path.resolve()}")
    print(f"Reduced figure:  {reduced_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
