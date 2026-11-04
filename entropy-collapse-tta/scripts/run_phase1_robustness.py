"""Phase 1 Robustness and Invariance Investigation.

Executes:
1. Horizon Sensitivity: num_steps in [500, 2000, 10000, 50000] with eta=0.05 across 5 seeds.
   Assesses whether critical delta* remains positive or drifts toward 0.
2. Flow-Time Invariance: Compares eta in {0.01, 0.05, 0.20} at matching flow-times T_flow = eta * steps.
3. Convergence & Gradient Norms: Tracks ||grad_b||_2 over time to verify convergence to stationary points.
4. Definition Robustness: Compares critical delta* under 3 collapse definitions (top_frac>0.9, H<0.3lnK, H<0.1lnK).
5. Uses sample standard deviation (ddof=1) throughout.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.theory.data import generate_gaussian_mixture
from src.theory.model import LinearSoftmaxModel
from src.theory.dynamics import simulate_dynamics, simulate_bias_dynamics_fast
from src.theory.basin import find_critical_delta_bisection, sweep_bias_basin
from src.utils.seed import seed_everything
from src.utils.logging import setup_logger, MetricLogger


def compute_sample_stats(values: List[float]) -> Tuple[float, float]:
    """Compute mean and sample standard deviation with ddof=1."""
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean_val, std_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 Robustness Experiments")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save logs and plots")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--separation", type=float, default=1.0)
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument("--samples_per_class", type=int, default=400)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("phase1_robustness", log_dir=str(logs_dir), filename="phase1_robustness.log")
    logger.info("=" * 60)
    logger.info("Starting Phase 1 Robustness & Dynamical Sensitivity Suite")
    logger.info("=" * 60)

    seeds = [42, 43, 44, 45, 46]
    base_eta = 0.05
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # -------------------------------------------------------------
    # Experiment 1: Horizon Sensitivity (num_steps in [500, 2000, 10000, 50000])
    # -------------------------------------------------------------
    logger.info("\n--- Experiment 1: Horizon Sensitivity ---")
    steps_list = [500, 2000, 10000, 50000]
    horizon_results: Dict[int, Dict[str, Any]] = {}

    for n_steps in steps_list:
        logger.info(f"Evaluating horizon num_steps = {n_steps} (eta={base_eta}, T_flow={base_eta * n_steps:.1f})...")
        crit_deltas_for_horizon: List[float] = []

        for s in seeds:
            Xs, ys, cents = generate_gaussian_mixture(
                num_classes=args.num_classes,
                dim=args.dim,
                samples_per_class=args.samples_per_class,
                separation=args.separation,
                noise_std=args.noise_std,
                seed=s,
            )
            m = LinearSoftmaxModel(num_classes=args.num_classes, dim=args.dim)
            m.fit_centroid_solution(cents, scale=1.0)

            crit_d = find_critical_delta_bisection(
                model_template=m,
                X=Xs,
                y=ys,
                delta_low=0.0,
                delta_high=4.0,
                tol=0.005,
                max_iter=14,
                num_steps=n_steps,
                lr=base_eta,
                adapt_mode="bias_only",
                collapse_definition="standard",
                fast=True,
            )
            crit_deltas_for_horizon.append(crit_d)

        mean_d, std_d = compute_sample_stats(crit_deltas_for_horizon)
        horizon_results[n_steps] = {
            "per_seed": crit_deltas_for_horizon,
            "mean": mean_d,
            "std_ddof1": std_d,
            "flow_time": base_eta * n_steps,
        }
        logger.info(f"Horizon {n_steps:5d} steps -> Critical delta* = {mean_d:.4f} +/- {std_d:.4f} (ddof=1)")

    # Plot 1: Horizon Sensitivity: delta* vs num_steps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    means = [horizon_results[s]["mean"] for s in steps_list]
    stds = [horizon_results[s]["std_ddof1"] for s in steps_list]

    # Linear scale
    ax1.errorbar(steps_list, means, yerr=stds, fmt="o-", capsize=5, color="#1f77b4", linewidth=2.0, markersize=7)
    ax1.set_xlabel("Optimization Horizon (num_steps)", fontweight="bold")
    ax1.set_ylabel(r"Critical Initial Bias $\delta^*$", fontweight="bold")
    ax1.set_title(r"Horizon Sensitivity of Critical Bias $\delta^*$", fontweight="bold")
    ax1.set_ylim(bottom=0.0)

    # Log scale in steps
    ax2.errorbar(steps_list, means, yerr=stds, fmt="s-", capsize=5, color="#d62728", linewidth=2.0, markersize=7)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Optimization Horizon $\log_{10}(\mathrm{num\_steps})$", fontweight="bold")
    ax2.set_ylabel(r"Critical Initial Bias $\delta^*$", fontweight="bold")
    ax2.set_title(r"Log-Scale Horizon Behavior", fontweight="bold")
    ax2.set_ylim(bottom=0.0)

    plt.tight_layout()
    fig_path1 = figures_dir / "phase1_horizon_sensitivity.png"
    fig.savefig(fig_path1, dpi=300)
    plt.close(fig)
    logger.info(f"Saved horizon sensitivity plot to: {fig_path1}")

    # -------------------------------------------------------------
    # Experiment 2: Flow-Time Invariance Check (eta in {0.01, 0.05, 0.20})
    # -------------------------------------------------------------
    logger.info("\n--- Experiment 2: Flow-Time Invariance Check ---")
    # Test at two matching flow-times T_flow = 25 and T_flow = 100
    target_flow_times = [25.0, 100.0]
    etas = [0.01, 0.05, 0.20]
    flow_results: Dict[float, Dict[float, Dict[str, float]]] = {}

    for T_flow in target_flow_times:
        flow_results[T_flow] = {}
        logger.info(f"Checking invariance at continuous flow-time T_flow = {T_flow:.1f}:")
        for eta in etas:
            steps = int(round(T_flow / eta))
            deltas_seed: List[float] = []
            for s in seeds:
                Xs, ys, cents = generate_gaussian_mixture(
                    num_classes=args.num_classes,
                    dim=args.dim,
                    samples_per_class=args.samples_per_class,
                    separation=args.separation,
                    noise_std=args.noise_std,
                    seed=s,
                )
                m = LinearSoftmaxModel(num_classes=args.num_classes, dim=args.dim)
                m.fit_centroid_solution(cents, scale=1.0)
                crit_d = find_critical_delta_bisection(
                    model_template=m,
                    X=Xs,
                    y=ys,
                    delta_low=0.0,
                    delta_high=4.0,
                    tol=0.005,
                    max_iter=14,
                    num_steps=steps,
                    lr=eta,
                    adapt_mode="bias_only",
                    fast=True,
                )
                deltas_seed.append(crit_d)

            m_val, s_val = compute_sample_stats(deltas_seed)
            flow_results[T_flow][eta] = {"steps": steps, "mean": m_val, "std_ddof1": s_val}
            logger.info(f"  T_flow={T_flow:5.1f} | eta={eta:.2f} (steps={steps:5d}) -> delta* = {m_val:.4f} +/- {s_val:.4f}")

    # Plot 2: Flow-time invariance
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors_flow = ["#2ca02c", "#9467bd"]
    for i, T_flow in enumerate(target_flow_times):
        sub_means = [flow_results[T_flow][eta]["mean"] for eta in etas]
        sub_stds = [flow_results[T_flow][eta]["std_ddof1"] for eta in etas]
        ax.errorbar(
            etas,
            sub_means,
            yerr=sub_stds,
            fmt="o--",
            capsize=5,
            color=colors_flow[i],
            linewidth=2.2,
            markersize=8,
            label=rf"$T_{{\mathrm{{flow}}}} = \eta \cdot N = {T_flow:.0f}$",
        )

    ax.set_xlabel(r"Learning Rate (Step Size) $\eta$", fontweight="bold")
    ax.set_ylabel(r"Critical Initial Bias $\delta^*$", fontweight="bold")
    ax.set_title(r"Flow-Time Invariance: $\delta^*$ Across Step Sizes $\eta$", fontweight="bold")
    ax.set_xticks(etas)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig_path2 = figures_dir / "phase1_flow_time_check.png"
    fig.savefig(fig_path2, dpi=300)
    plt.close(fig)
    logger.info(f"Saved flow-time invariance plot to: {fig_path2}")

    # -------------------------------------------------------------
    # Experiment 3: Convergence & Gradient Norm Dynamics
    # -------------------------------------------------------------
    logger.info("\n--- Experiment 3: Convergence & Gradient Norm Dynamics ---")
    conv_steps = 10000
    test_deltas = [0.0, 0.3, 0.6, 0.9, 1.2, 1.8]
    conv_results: Dict[float, Dict[str, Any]] = {}

    X_conv, y_conv, cents_conv = generate_gaussian_mixture(
        num_classes=args.num_classes,
        dim=args.dim,
        samples_per_class=args.samples_per_class,
        separation=args.separation,
        noise_std=args.noise_std,
        seed=42,
    )
    m_conv = LinearSoftmaxModel(num_classes=args.num_classes, dim=args.dim)
    m_conv.fit_centroid_solution(cents_conv, scale=1.0)
    with torch.no_grad():
        F_conv = X_conv @ m_conv.weight.T

    for delta in test_deltas:
        b_init = m_conv.bias.clone()
        b_init[0] += delta
        res_conv = simulate_bias_dynamics_fast(
            precomputed_F=F_conv,
            initial_bias=b_init,
            y_eval=y_conv,
            learning_rate=base_eta,
            num_steps=conv_steps,
            collapse_definition="standard",
            log_every=50,
        )
        final_g = res_conv["final_grad_norm"]
        conv_results[delta] = {
            "steps": res_conv["steps"],
            "grad_norm": res_conv["grad_norm_history"],
            "marginal_entropy": res_conv["marginal_entropy"],
            "final_grad_norm": final_g,
            "is_collapsed": res_conv["is_collapsed"],
            "collapse_step": res_conv["collapse_step"],
            "final_h_pbar": res_conv["final_marginal_entropy"],
            "final_top_fraction": res_conv["final_top_fraction"],
        }
        logger.info(
            f"delta={delta:4.1f} | Final ||grad_b||={final_g:.2e} | "
            f"Collapsed={str(res_conv['is_collapsed']):<5} | "
            f"Final H(p_bar)={res_conv['final_marginal_entropy']:.4f}"
        )

    # Plot 3: Gradient Norm Convergence over time (log scale)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    colors_conv = sns.color_palette("tab10", len(test_deltas))

    for i, delta in enumerate(test_deltas):
        rec = conv_results[delta]
        lbl = rf"$\delta = {delta:.1f}$" + (" [COLLAPSED]" if rec["is_collapsed"] else " [STABLE]")
        ax1.plot(rec["steps"], rec["grad_norm"], label=lbl, color=colors_conv[i], linewidth=2.0)
        ax2.plot(rec["steps"], rec["marginal_entropy"], label=lbl, color=colors_conv[i], linewidth=2.0)

    ax1.set_yscale("log")
    ax1.set_xlabel("Steps (up to 10,000)", fontweight="bold")
    ax1.set_ylabel(r"Gradient Norm $\|\nabla_b \mathcal{L}\|_2$ (log-scale)", fontweight="bold")
    ax1.set_title(r"Gradient Norm Dynamics $\|\nabla_b \mathcal{L}\|_2$", fontweight="bold")
    ax1.legend(frameon=True)

    ax2.set_xlabel("Steps (up to 10,000)", fontweight="bold")
    ax2.set_ylabel(r"Marginal Entropy $H(\bar{p})$ [nats]", fontweight="bold")
    ax2.set_title(r"Long-Horizon Marginal Diversity Trajectories", fontweight="bold")
    ax2.axhline(np.log(args.num_classes), color="gray", linestyle="--", label="Uniform Max ln(3)")
    ax2.legend(frameon=True)

    plt.tight_layout()
    fig_path3 = figures_dir / "phase1_gradient_norm_convergence.png"
    fig.savefig(fig_path3, dpi=300)
    plt.close(fig)
    logger.info(f"Saved gradient norm convergence plot to: {fig_path3}")

    # -------------------------------------------------------------
    # Experiment 4: Definition Robustness Check (T=500 and T=2000)
    # -------------------------------------------------------------
    logger.info("\n--- Experiment 4: Collapse Definition Robustness ---")
    definitions = [
        ("def_a", r"Def A: Top-Class $\ge 0.90$"),
        ("def_b", r"Def B: $H(\bar{p}) < 0.3 \ln K$"),
        ("standard", r"Standard: Def A & Def B"),
        ("def_c", r"Def C: $H(\bar{p}) < 0.1 \ln K$"),
    ]
    def_results_500: Dict[str, Dict[str, Any]] = {}
    def_results_2000: Dict[str, Dict[str, Any]] = {}

    for def_key, def_label in definitions:
        # T = 500
        crit_500: List[float] = []
        for s in seeds:
            Xs, ys, cents = generate_gaussian_mixture(
                num_classes=args.num_classes,
                dim=args.dim,
                samples_per_class=args.samples_per_class,
                separation=args.separation,
                noise_std=args.noise_std,
                seed=s,
            )
            m = LinearSoftmaxModel(num_classes=args.num_classes, dim=args.dim)
            m.fit_centroid_solution(cents, scale=1.0)
            c = find_critical_delta_bisection(
                m, Xs, ys, 0.0, 4.0, tol=0.005, max_iter=14, num_steps=500, lr=base_eta,
                collapse_definition=def_key, fast=True,
            )
            crit_500.append(c)

        m500, s500 = compute_sample_stats(crit_500)
        def_results_500[def_key] = {"label": def_label, "per_seed": crit_500, "mean": m500, "std_ddof1": s500}
        logger.info(f"{def_label:<45} (T=500)  -> delta* = {m500:.4f} +/- {s500:.4f} (ddof=1)")

        # T = 2000
        crit_2000: List[float] = []
        for s in seeds:
            Xs, ys, cents = generate_gaussian_mixture(
                num_classes=args.num_classes,
                dim=args.dim,
                samples_per_class=args.samples_per_class,
                separation=args.separation,
                noise_std=args.noise_std,
                seed=s,
            )
            m = LinearSoftmaxModel(num_classes=args.num_classes, dim=args.dim)
            m.fit_centroid_solution(cents, scale=1.0)
            c = find_critical_delta_bisection(
                m, Xs, ys, 0.0, 4.0, tol=0.005, max_iter=14, num_steps=2000, lr=base_eta,
                collapse_definition=def_key, fast=True,
            )
            crit_2000.append(c)

        m2000, s2000 = compute_sample_stats(crit_2000)
        def_results_2000[def_key] = {"label": def_label, "per_seed": crit_2000, "mean": m2000, "std_ddof1": s2000}
        logger.info(f"{def_label:<45} (T=2000) -> delta* = {m2000:.4f} +/- {s2000:.4f} (ddof=1)")

    # Plot 4: Definition Robustness Comparison across Horizons
    fig, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(14, 5.0))
    def_keys = [d[0] for d in definitions]
    labels_clean = ["Def A (Top>0.9)", r"Def B ($H < 0.3\ln K$)", "Standard (A & B)", r"Def C ($H < 0.1\ln K$)"]
    colors_bar = ["#3274a1", "#e1812c", "#c03d3e", "#3a923a"]

    # Subplot A: T=500
    means_500 = [def_results_500[k]["mean"] for k in def_keys]
    stds_500 = [def_results_500[k]["std_ddof1"] for k in def_keys]
    bars1 = ax_d1.bar(labels_clean, means_500, yerr=stds_500, capsize=6, color=colors_bar, width=0.55)
    for bar, val in zip(bars1, means_500):
        ax_d1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.08, f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
    ax_d1.set_ylabel(r"Critical Initial Bias $\delta^*$", fontweight="bold")
    ax_d1.set_title(r"Finite Horizon ($T=500$ steps): Metastable Thresholds", fontweight="bold")
    ax_d1.set_ylim(0, max(means_500) * 1.25)

    # Subplot B: T=2000
    means_2000 = [def_results_2000[k]["mean"] for k in def_keys]
    stds_2000 = [def_results_2000[k]["std_ddof1"] for k in def_keys]
    bars2 = ax_d2.bar(labels_clean, means_2000, yerr=stds_2000, capsize=6, color=colors_bar, width=0.55)
    for bar, val in zip(bars2, means_2000):
        ax_d2.text(bar.get_x() + bar.get_width() / 2.0, 0.05, f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
    ax_d2.set_ylabel(r"Critical Initial Bias $\delta^*$", fontweight="bold")
    ax_d2.set_title(r"Long Horizon ($T=2000$ steps): Universal Collapse ($\delta^* \to 0$)", fontweight="bold")
    ax_d2.set_ylim(0, 1.0)

    plt.tight_layout()
    fig_path4 = figures_dir / "phase1_definition_robustness.png"
    fig.savefig(fig_path4, dpi=300)
    plt.close(fig)
    logger.info(f"Saved definition robustness plot to: {fig_path4}")

    # -------------------------------------------------------------
    # 5. Export JSON summary
    # -------------------------------------------------------------
    summary_robustness = {
        "experiment": "phase1_robustness_suite",
        "horizon_sensitivity": {
            str(k): {
                "mean": round(v["mean"], 4),
                "std_ddof1": round(v["std_ddof1"], 4),
                "flow_time": v["flow_time"],
                "per_seed": [round(float(x), 4) for x in v["per_seed"]],
            }
            for k, v in horizon_results.items()
        },
        "flow_time_check": {
            str(T_flow): {
                str(eta): {
                    "steps": v["steps"],
                    "mean": round(v["mean"], 4),
                    "std_ddof1": round(v["std_ddof1"], 4),
                }
                for eta, v in sub_dict.items()
            }
            for T_flow, sub_dict in flow_results.items()
        },
        "convergence_check": {
            str(d): {
                "final_grad_norm": f"{rec['final_grad_norm']:.4e}",
                "is_collapsed": rec["is_collapsed"],
                "final_h_pbar": round(rec["final_h_pbar"], 4),
                "final_top_fraction": round(rec["final_top_fraction"], 4),
                "collapse_step": rec["collapse_step"],
            }
            for d, rec in conv_results.items()
        },
        "definition_robustness_T500": {
            k: {
                "label": v["label"],
                "mean": round(v["mean"], 4),
                "std_ddof1": round(v["std_ddof1"], 4),
                "per_seed": [round(float(x), 4) for x in v["per_seed"]],
            }
            for k, v in def_results_500.items()
        },
        "definition_robustness_T2000": {
            k: {
                "label": v["label"],
                "mean": round(v["mean"], 4),
                "std_ddof1": round(v["std_ddof1"], 4),
                "per_seed": [round(float(x), 4) for x in v["per_seed"]],
            }
            for k, v in def_results_2000.items()
        },
        "saved_figures": [
            str(fig_path1),
            str(fig_path2),
            str(fig_path3),
            str(fig_path4),
        ],
    }

    summary_file = logs_dir / "phase1_robustness_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_robustness, f, indent=2)
    logger.info(f"Saved robustness summary JSON to: {summary_file}")
    logger.info("All robustness checks completed successfully!")


if __name__ == "__main__":
    main()
