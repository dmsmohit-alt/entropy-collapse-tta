"""Run Phase 1 theoretical toy model experiments.

Executes:
1. Entropy minimization dynamics across varying initial bias offsets delta.
2. Generates H(p_bar) and bias trajectory plots.
3. Maps the basin of attraction boundary (critical delta*) over multiple random seeds.
4. Generates a 2D phase diagram of separation vs initial bias offset.
5. Saves all figures, CSV metric logs, and summary JSON.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path so src is always found
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.theory.data import generate_gaussian_mixture
from src.theory.model import LinearSoftmaxModel
from src.theory.dynamics import simulate_dynamics, is_collapsed_state
from src.theory.basin import (
    find_critical_delta_bisection,
    sweep_bias_basin,
    compute_phase_diagram_2d,
)
from src.utils.seed import seed_everything
from src.utils.logging import setup_logger, MetricLogger
from src.utils.config import load_yaml_config, save_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 Toy Model Experiments")
    parser.add_argument("--config", type=str, default="configs/theory_toy.yaml", help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save logs and plots")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("phase1_theory", log_dir=str(logs_dir), filename="phase1_theory.log")
    logger.info("Starting Phase 1 Toy Model Experiments...")

    base_seed = config.get("experiment", {}).get("seed", 42)
    seed_everything(base_seed)

    # Model & Data Configuration
    num_classes = config.get("model", {}).get("num_classes", 3)
    dim = config.get("model", {}).get("dim", 2)
    weight_scale = config.get("model", {}).get("weight_scale", 1.0)
    separation = config.get("data", {}).get("class_means_radius", 1.0)
    noise_std = config.get("data", {}).get("cluster_std", 1.0)
    samples_per_class = config.get("data", {}).get("num_samples_per_class", 400)
    lr = config.get("dynamics", {}).get("learning_rate", 0.05)
    num_steps = config.get("dynamics", {}).get("num_steps", 400)
    deltas_to_test = config.get("dynamics", {}).get("deltas_to_test", [0.0, 0.3, 0.6, 0.9, 1.2, 1.8, 2.5])
    seeds = config.get("dynamics", {}).get("seeds", [42, 43, 44, 45, 46])

    logger.info(f"Generating Gaussian mixture: K={num_classes}, dim={dim}, separation={separation}, scale={weight_scale}")
    X, y, centroids = generate_gaussian_mixture(
        num_classes=num_classes,
        dim=dim,
        samples_per_class=samples_per_class,
        separation=separation,
        noise_std=noise_std,
        seed=base_seed,
    )

    base_model = LinearSoftmaxModel(num_classes=num_classes, dim=dim)
    base_model.fit_centroid_solution(centroids, scale=weight_scale)

    # -------------------------------------------------------------
    # 1. Trajectory comparison across delta values
    # -------------------------------------------------------------
    trajectories: Dict[float, Dict] = {}
    logger.info(f"Simulating dynamics for deltas: {deltas_to_test}")

    # Track metrics with MetricLogger for persistent CSV record
    metric_logger = MetricLogger(output_dir=str(logs_dir), experiment_name="phase1_trajectories")

    for delta in deltas_to_test:
        m = base_model.clone()
        m.apply_bias_offset(delta=delta, class_idx=0)
        res = simulate_dynamics(
            m, X, y, learning_rate=lr, num_steps=num_steps, adapt_mode="bias_only", objective="tent"
        )
        trajectories[delta] = res
        logger.info(
            f"delta={delta:4.1f} | Collapsed={str(res['is_collapsed']):<5} | "
            f"Final H(p_bar)={res['final_marginal_entropy']:.4f} | "
            f"Final Top Frac={res['final_top_fraction']:.4f} | "
            f"Collapse Step={res['collapse_step']}"
        )
        for s, h, top in zip(res["steps"], res["marginal_entropy"], res["top_class_fraction"]):
            metric_logger.log_step(int(s), {f"h_pbar_delta_{delta}": h, f"top_frac_delta_{delta}": top})

    metric_logger.save_csv()

    # Plot 1: Marginal Entropy Trajectories H(p_bar) vs Steps
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    max_entropy = np.log(num_classes)
    ax.axhline(max_entropy, color="gray", linestyle="--", alpha=0.8, label=f"ln({num_classes}) = {max_entropy:.3f} (Uniform)")
    ax.axhline(0.3 * max_entropy, color="red", linestyle=":", alpha=0.8, label="Collapse Cutoff (0.3 ln K)")

    colors = sns.color_palette("tab10", len(deltas_to_test))
    for i, delta in enumerate(deltas_to_test):
        res = trajectories[delta]
        lbl = rf"$\delta = {delta:.1f}$" + (" [COLLAPSED]" if res["is_collapsed"] else " [STABLE]")
        ax.plot(res["steps"], res["marginal_entropy"], label=lbl, color=colors[i], linewidth=2.2)

    ax.set_xlabel("Adaptation Steps (Gradient Descent)", fontweight="bold")
    ax.set_ylabel(r"Marginal Entropy $H(\bar{p})$ [nats]", fontweight="bold")
    ax.set_title(rf"Marginal Entropy Trajectories under TENT (K={num_classes}, $R={separation:.1f}$)", fontweight="bold", pad=12)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    fig_path1 = figures_dir / "phase1_h_pbar_trajectories.png"
    fig.savefig(fig_path1, dpi=300)
    plt.close(fig)
    logger.info(f"Saved trajectory plot to: {fig_path1}")

    # Plot 2: Bias Dynamics: b_0(t) vs b_k(t)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    # Subplot A: Stable run (delta = 0.3)
    res_stable = trajectories[0.3]
    for k in range(num_classes):
        ax1.plot(res_stable["steps"], res_stable["bias_history"][:, k], label=rf"$b_{k}(t)$", linewidth=2.2)
    ax1.set_title(r"Stable Regime ($\delta = 0.3$)", fontweight="bold")
    ax1.set_xlabel("Steps", fontweight="bold")
    ax1.set_ylabel("Bias Value", fontweight="bold")
    ax1.legend()

    # Subplot B: Collapsing run (delta = 1.2)
    res_collapsing = trajectories[1.2]
    for k in range(num_classes):
        ax2.plot(res_collapsing["steps"], res_collapsing["bias_history"][:, k], label=rf"$b_{k}(t)$", linewidth=2.2)
    ax2.set_title(r"Collapse Regime: Rich-Get-Richer ($\delta = 1.2$)", fontweight="bold")
    ax2.set_xlabel("Steps", fontweight="bold")
    ax2.legend()
    plt.tight_layout()
    fig_path2 = figures_dir / "phase1_bias_dynamics.png"
    fig.savefig(fig_path2, dpi=300)
    plt.close(fig)
    logger.info(f"Saved bias dynamics plot to: {fig_path2}")

    # -------------------------------------------------------------
    # 2. Critical delta* Bisection across Multiple Random Seeds
    # -------------------------------------------------------------
    bisect_cfg = config.get("dynamics", {}).get("bisection", {})
    d_low = bisect_cfg.get("delta_low", 0.0)
    d_high = bisect_cfg.get("delta_high", 4.0)
    tol = bisect_cfg.get("tol", 0.005)
    max_iter = bisect_cfg.get("max_iter", 14)

    critical_deltas: List[float] = []
    logger.info(f"Estimating critical delta* via bisection across seeds: {seeds}")

    for s in seeds:
        Xs, ys, cents = generate_gaussian_mixture(
            num_classes=num_classes,
            dim=dim,
            samples_per_class=samples_per_class,
            separation=separation,
            noise_std=noise_std,
            seed=s,
        )
        m_s = LinearSoftmaxModel(num_classes=num_classes, dim=dim)
        m_s.fit_centroid_solution(cents, scale=weight_scale)

        crit_d = find_critical_delta_bisection(
            model_template=m_s,
            X=Xs,
            y=ys,
            delta_low=d_low,
            delta_high=d_high,
            tol=tol,
            max_iter=max_iter,
            num_steps=num_steps,
            lr=lr,
            adapt_mode="bias_only",
        )
        critical_deltas.append(crit_d)
        logger.info(f"Seed {s:2d} -> Critical delta* = {crit_d:.4f}")

    mean_crit_delta = float(np.mean(critical_deltas))
    std_crit_delta = float(np.std(critical_deltas))
    logger.info(f"Critical delta* estimate: {mean_crit_delta:.4f} +/- {std_crit_delta:.4f}")

    # -------------------------------------------------------------
    # 3. 1D Basin Grid Sweep & Plot
    # -------------------------------------------------------------
    s1d_cfg = config.get("dynamics", {}).get("sweep_1d", {})
    fine_deltas = np.linspace(
        s1d_cfg.get("min_delta", 0.0),
        s1d_cfg.get("max_delta", 3.0),
        s1d_cfg.get("num_points", 31),
    )
    logger.info(f"Running fine 1D basin sweep over {len(fine_deltas)} delta points...")
    sweep_results = sweep_bias_basin(
        model_template=base_model,
        X=X,
        y=y,
        delta_grid=fine_deltas,
        learning_rate=lr,
        num_steps=num_steps,
        adapt_mode="bias_only",
        objective="tent",
    )

    # Plot 3: 1D Basin curve
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(fine_deltas, sweep_results["final_h_pbar"], "o-", color="#1f77b4", markersize=5, linewidth=2.0, label=r"Final $H(\bar{p})$")
    ax.axvline(mean_crit_delta, color="#d62728", linestyle="--", linewidth=2.0, label=rf"Critical $\delta^* = {mean_crit_delta:.3f} \pm {std_crit_delta:.3f}$")
    ax.axhline(0.3 * max_entropy, color="gray", linestyle=":", linewidth=1.5, label="Collapse Threshold (0.3 ln K)")
    ax.set_xlabel(r"Initial Bias Offset $\delta$ on Class 0", fontweight="bold")
    ax.set_ylabel(r"Final Marginal Entropy $H(\bar{p})$ [nats]", fontweight="bold")
    ax.set_title(r"Basin Boundary: Final Diversity vs Initial Bias $\delta$", fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig_path3 = figures_dir / "phase1_basin_1d.png"
    fig.savefig(fig_path3, dpi=300)
    plt.close(fig)
    logger.info(f"Saved 1D basin curve to: {fig_path3}")

    # -------------------------------------------------------------
    # 4. 2D Phase Diagram (Separation vs Bias Offset)
    # -------------------------------------------------------------
    p2d_cfg = config.get("dynamics", {}).get("phase_diagram_2d", {})
    separations = np.linspace(
        p2d_cfg.get("min_sep", 0.6),
        p2d_cfg.get("max_sep", 2.0),
        p2d_cfg.get("num_sep", 8),
    )
    grid_deltas = np.linspace(
        p2d_cfg.get("min_delta", 0.0),
        p2d_cfg.get("max_delta", 3.0),
        p2d_cfg.get("num_delta", 10),
    )
    p2d_steps = p2d_cfg.get("steps", 300)

    logger.info(f"Computing 2D phase diagram over {len(separations)}x{len(grid_deltas)} points...")
    phase_res = compute_phase_diagram_2d(
        separations=separations,
        deltas=grid_deltas,
        num_classes=num_classes,
        dim=dim,
        samples_per_class=250,
        noise_std=noise_std,
        learning_rate=lr,
        num_steps=p2d_steps,
        seed=base_seed,
    )

    # Plot 4: 2D Heatmap
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    grid = phase_res["phase_grid"].astype(int)  # 1 = collapsed, 0 = stable
    sns.heatmap(
        grid,
        xticklabels=np.round(grid_deltas, 2),
        yticklabels=np.round(separations, 2),
        cmap=["#2ca02c", "#d62728"],  # green = stable, red = collapsed
        cbar_kws={"ticks": [0.25, 0.75], "label": "Adaptation State"},
        ax=ax,
        linewidths=0.5,
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.set_yticklabels(["Stable (0)", "Collapsed (1)"])
    ax.set_xlabel(r"Initial Bias $\delta$", fontweight="bold")
    ax.set_ylabel(r"Class Centroid Radius $R$ (Separation)", fontweight="bold")
    ax.set_title(r"2D Phase Diagram: Stability vs Collapse Basin", fontweight="bold")
    plt.tight_layout()
    fig_path4 = figures_dir / "phase1_basin_phase_diagram_2d.png"
    fig.savefig(fig_path4, dpi=300)
    plt.close(fig)
    logger.info(f"Saved 2D phase diagram to: {fig_path4}")

    # -------------------------------------------------------------
    # 5. Save Experiment Config & Summary JSON
    # -------------------------------------------------------------
    summary_data = {
        "experiment": "phase1_toy_model",
        "num_classes": num_classes,
        "dim": dim,
        "separation": separation,
        "noise_std": noise_std,
        "learning_rate": lr,
        "num_steps": num_steps,
        "deltas_tested": deltas_to_test,
        "seeds_tested": seeds,
        "critical_deltas_per_seed": {str(s): round(float(d), 4) for s, d in zip(seeds, critical_deltas)},
        "critical_delta_mean": round(mean_crit_delta, 4),
        "critical_delta_std": round(std_crit_delta, 4),
        "saved_figures": [
            str(fig_path1),
            str(fig_path2),
            str(fig_path3),
            str(fig_path4),
        ],
    }

    summary_file = logs_dir / "phase1_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved summary JSON to: {summary_file}")

    save_yaml_config(config, logs_dir / "phase1_config.yaml")
    logger.info("Phase 1 experiments completed successfully!")


if __name__ == "__main__":
    main()
