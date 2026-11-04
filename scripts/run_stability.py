"""Phase 2: verify the local stability analysis numerically.

The script does not replace the derivation.  It checks four derived claims on
saved, reproducible artifacts:

1. The analytic entropy and information gradients match autograd.
2. The symmetric-point bias-block Hessian equals ``(lambda - 1) C``.
3. With centred features, the only positive Hessian eigenvalue is the marginal
   mode ``(lambda - 1) / K``.
4. The reduced ``W = 0`` bias flow contracts for ``lambda > 1`` and expands for
   ``lambda < 1``, with the predicted one-step factor.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402

from theory.basin import validate_seeds, write_json  # noqa: E402
from theory.stability import (  # noqa: E402
    StabilitySpec,
    discrete_step_bound,
    entropy_logit_gradient,
    information_logit_gradient,
    information_objective,
    marginal_mode_curvature,
    mean_predictive_entropy,
    parameter_gradients,
    parameter_hessian,
    simulate_reduced_bias,
    symmetric_b_block_hessian,
)
from utils.config import config_hash, load_config, validate_run_config  # noqa: E402
from utils.logging import JsonlLogger, configure_console_logging, utc_now  # noqa: E402
from utils.run import create_run_dir  # noqa: E402
from utils.seeding import seed_everything  # noqa: E402

DTYPE = torch.float64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "theory_phase2.yaml",
        help="Phase 2 YAML config.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override; may be repeated.",
    )
    return parser.parse_args()


def _random_features(spec: StabilitySpec, *, seed: int) -> torch.Tensor:
    """Return zero-mean features so ``W`` cannot excite the common logit mode."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(spec.num_samples, spec.dim, generator=generator, dtype=DTYPE)
    return x - x.mean(dim=0, keepdim=True)


def _random_logits(spec: StabilitySpec, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(spec.num_samples, spec.num_classes, generator=generator, dtype=DTYPE)


def _gradient_checks(spec: StabilitySpec, *, seed: int) -> list[dict[str, Any]]:
    """Compare analytic logit and parameter gradients with autograd."""
    base_logits = _random_logits(spec, seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(spec.num_samples, spec.dim, generator=generator, dtype=DTYPE)
    weight_base = torch.randn(spec.num_classes, spec.dim, generator=generator, dtype=DTYPE)
    bias_base = torch.randn(spec.num_classes, generator=generator, dtype=DTYPE)

    rows: list[dict[str, Any]] = []
    for lam in spec.lambda_values:
        logits = base_logits.clone().requires_grad_(True)
        autograd_entropy = torch.autograd.grad(mean_predictive_entropy(logits), logits)[0]
        analytic_entropy = entropy_logit_gradient(logits.detach())

        logits = base_logits.clone().requires_grad_(True)
        autograd_info = torch.autograd.grad(information_objective(logits, lam), logits)[0]
        analytic_info = information_logit_gradient(logits.detach(), lam)

        weight = weight_base.clone().requires_grad_(True)
        bias = bias_base.clone().requires_grad_(True)
        loss = information_objective(x @ weight.T + bias, lam)
        autograd_weight, autograd_bias = torch.autograd.grad(loss, [weight, bias])
        analytic_weight, analytic_bias = parameter_gradients(
            x, weight_base, bias_base, lam
        )

        rows.append(
            {
                "seed": seed,
                "lambda": lam,
                "entropy_logit_max_abs_error": float(
                    (autograd_entropy - analytic_entropy).abs().max()
                ),
                "info_logit_max_abs_error": float(
                    (autograd_info - analytic_info).abs().max()
                ),
                "entropy_weight_max_abs_error": float(
                    (autograd_weight - analytic_weight).abs().max()
                ),
                "entropy_bias_max_abs_error": float(
                    (autograd_bias - analytic_bias).abs().max()
                ),
                "logit_gradient_sum_max_abs": float(analytic_info.sum(dim=1).abs().max()),
            }
        )
    return rows


def _hessian_rows(
    spec: StabilitySpec,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], float, float]:
    """Check the symmetric-point Hessian and return rows plus max errors."""
    x = _random_features(spec, seed=seed)
    num_classes = spec.num_classes
    dim = spec.dim
    rows: list[dict[str, Any]] = []
    max_b_block_error = 0.0
    max_full_max_error = 0.0

    for lam in spec.lambda_values:
        weight = torch.zeros(num_classes, dim, dtype=DTYPE, requires_grad=True)
        bias = torch.zeros(num_classes, dtype=DTYPE, requires_grad=True)
        hessian = parameter_hessian(x, weight, bias, lam)
        b_block = hessian[num_classes * dim :, num_classes * dim :]
        expected_b_block = symmetric_b_block_hessian(lam, num_classes)
        b_block_error = float((b_block - expected_b_block).abs().max())
        max_b_block_error = max(max_b_block_error, b_block_error)

        b_eigenvalues = torch.linalg.eigvalsh(b_block)
        predicted = torch.sort(
            torch.tensor(
                [(lam - 1.0) / num_classes] * (num_classes - 1) + [0.0],
                dtype=DTYPE,
            )
        ).values
        full_eigenvalues = torch.linalg.eigvalsh(hessian)
        full_max = float(full_eigenvalues.max())
        predicted_full_max = max(0.0, marginal_mode_curvature(lam, num_classes))
        full_max_error = abs(full_max - predicted_full_max)
        max_full_max_error = max(max_full_max_error, full_max_error)

        row: dict[str, Any] = {
            "seed": seed,
            "lambda": lam,
            "marginal_curvature": marginal_mode_curvature(lam, num_classes),
            "step_bound": discrete_step_bound(lam, num_classes),
            "b_block_max_abs_error": b_block_error,
            "b_eigenvalue_max_abs_error": float((b_eigenvalues - predicted).abs().max()),
            "full_min_eigenvalue": float(full_eigenvalues.min()),
            "full_max_eigenvalue": full_max,
            "predicted_full_max": predicted_full_max,
            "full_max_abs_error": full_max_error,
        }
        for index, value in enumerate(b_eigenvalues.tolist()):
            row[f"b_eigenvalue_{index}"] = float(value)
        for index, value in enumerate(predicted.tolist()):
            row[f"predicted_b_eigenvalue_{index}"] = float(value)
        rows.append(row)

    return rows, max_b_block_error, max_full_max_error


def _write_hessian_csv(rows: list[dict[str, Any]], path: Path, num_classes: int) -> None:
    fieldnames = [
        "seed",
        "lambda",
        "marginal_curvature",
        "step_bound",
        "b_block_max_abs_error",
        "b_eigenvalue_max_abs_error",
        "full_min_eigenvalue",
        "full_max_eigenvalue",
        "predicted_full_max",
        "full_max_abs_error",
    ]
    fieldnames += [f"b_eigenvalue_{index}" for index in range(num_classes)]
    fieldnames += [f"predicted_b_eigenvalue_{index}" for index in range(num_classes)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _initial_bias(
    spec: StabilitySpec,
    *,
    epsilon: float,
    perturbation: list[float],
) -> torch.Tensor:
    if len(perturbation) != spec.num_classes:
        raise ValueError("reduced.perturbation must have one entry per class")
    vector = torch.tensor(perturbation, dtype=DTYPE)
    if not torch.isfinite(vector).all():
        raise ValueError("reduced.perturbation must be finite")
    if abs(float(vector.sum())) > 1e-12:
        raise ValueError("reduced.perturbation must sum to zero")
    if float(torch.linalg.vector_norm(vector)) == 0.0:
        raise ValueError("reduced.perturbation must be non-zero")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("reduced.initial_epsilon must be finite and > 0")
    return epsilon * vector


def _classify_reduced(
    initial_entropy: float,
    final_entropy: float,
    initial_spread: float,
    final_spread: float,
    *,
    num_classes: int,
) -> str:
    log_k = math.log(num_classes)
    if final_entropy < 0.5 * log_k and final_spread > initial_spread:
        return "collapse"
    if final_entropy > 0.95 * log_k and final_spread < initial_spread:
        return "balanced"
    if abs(final_entropy - initial_entropy) < 1e-12:
        return "neutral"
    return "partial"


def _run_reduced(
    spec: StabilitySpec,
    reduced_config: dict[str, Any],
    *,
    run_dir: Path,
) -> list[dict[str, Any]]:
    lambdas = tuple(float(value) for value in reduced_config["lambda_values"])
    initial = _initial_bias(
        spec,
        epsilon=float(reduced_config["initial_epsilon"]),
        perturbation=[float(value) for value in reduced_config["perturbation"]],
    )
    steps = int(reduced_config["steps"])
    learning_rate = float(reduced_config["learning_rate"])

    summaries: list[dict[str, Any]] = []
    with JsonlLogger(run_dir / "reduced_trajectories.jsonl") as logger:
        for lam in lambdas:
            records = simulate_reduced_bias(
                lam=lam,
                initial_bias=initial,
                steps=steps,
                learning_rate=learning_rate,
            )
            for record in records:
                logger.write({"lambda": lam, **record})
            initial_entropy = records[0]["marginal_entropy"]
            final_entropy = records[-1]["marginal_entropy"]
            initial_spread = records[0]["bias_spread"]
            final_spread = records[-1]["bias_spread"]
            summaries.append(
                {
                    "lambda": lam,
                    "initial_entropy": initial_entropy,
                    "final_entropy": final_entropy,
                    "initial_bias_spread": initial_spread,
                    "final_bias_spread": final_spread,
                    "entropy_change": final_entropy - initial_entropy,
                    "spread_ratio": (
                        final_spread / initial_spread if initial_spread > 0 else None
                    ),
                    "step_bound": discrete_step_bound(lam, spec.num_classes),
                    "behavior": _classify_reduced(
                        initial_entropy,
                        final_entropy,
                        initial_spread,
                        final_spread,
                        num_classes=spec.num_classes,
                    ),
                }
            )
    return summaries


def _run_step_size(
    spec: StabilitySpec,
    step_config: dict[str, Any],
) -> list[dict[str, Any]]:
    lam = float(step_config["lambda"])
    initial = _initial_bias(
        spec,
        epsilon=float(step_config["initial_epsilon"]),
        perturbation=[1.0] + [-1.0 / (spec.num_classes - 1)] * (spec.num_classes - 1),
    )
    steps = int(step_config["steps"])
    bound = discrete_step_bound(lam, spec.num_classes)
    rows: list[dict[str, Any]] = []
    for learning_rate in step_config["learning_rates"]:
        learning_rate = float(learning_rate)
        records = simulate_reduced_bias(
            lam=lam,
            initial_bias=initial,
            steps=steps,
            learning_rate=learning_rate,
        )
        observed = records[1]["bias_spread"] / records[0]["bias_spread"]
        predicted = 1.0 - learning_rate * (lam - 1.0) / spec.num_classes
        rows.append(
            {
                "lambda": lam,
                "learning_rate": learning_rate,
                "predicted_one_step_factor": predicted,
                "observed_one_step_factor": observed,
                "absolute_factor": abs(predicted),
                "stable_by_linearisation": abs(predicted) < 1.0,
                "within_bound": bound is not None and learning_rate < bound,
                "absolute_error": abs(observed - abs(predicted)),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    config = load_config(args.config, overrides=args.override)
    validate_run_config(config)
    configure_console_logging(str(config["logging"]["level"]))
    seed_everything(int(config["seed"]), deterministic=True)

    spec = StabilitySpec.from_mapping(config["stability"])
    tolerance = float(config["stability"].get("tolerance", 1e-9))
    verification_seeds = validate_seeds(
        config["stability"].get("verification_seeds", [0, 1, 2])
    )

    run = create_run_dir(config)
    start_time = time.perf_counter()
    with JsonlLogger(run.run_dir / "run_events.jsonl") as event_log:
        event_log.write(
            {
                "event": "start",
                "config_hash": config_hash(config),
                "stability_spec": asdict(spec),
                "verification_seeds": list(verification_seeds),
                "tolerance": tolerance,
            }
        )

        gradient_checks: list[dict[str, Any]] = []
        hessian_rows: list[dict[str, Any]] = []
        max_b_block_error = 0.0
        max_full_max_error = 0.0
        for seed in verification_seeds:
            gradient_checks.extend(_gradient_checks(spec, seed=seed))
            seed_rows, seed_b_block_error, seed_full_max_error = _hessian_rows(
                spec, seed=seed + 1
            )
            hessian_rows.extend(seed_rows)
            max_b_block_error = max(max_b_block_error, seed_b_block_error)
            max_full_max_error = max(max_full_max_error, seed_full_max_error)
        _write_hessian_csv(hessian_rows, run.run_dir / "hessian_spectrum.csv", spec.num_classes)
        write_json({"seed_rows": gradient_checks}, run.run_dir / "gradient_checks.json")

        reduced_summary = _run_reduced(
            spec,
            config["reduced"],
            run_dir=run.run_dir,
        )
        step_size_rows = _run_step_size(spec, config["step_size"])

        max_gradient_error = max(
            max(
                row["entropy_logit_max_abs_error"],
                row["info_logit_max_abs_error"],
                row["entropy_weight_max_abs_error"],
                row["entropy_bias_max_abs_error"],
                row["logit_gradient_sum_max_abs"],
            )
            for row in gradient_checks
        )
        max_b_eigenvalue_error = max(
            row["b_eigenvalue_max_abs_error"] for row in hessian_rows
        )
        max_step_error = max(row["absolute_error"] for row in step_size_rows)

        elapsed = time.perf_counter() - start_time
        summary = {
            "experiment": config["experiment"],
            "created_utc": utc_now(),
            "runtime_seconds": elapsed,
            "stability_spec": asdict(spec),
            "verification_seeds": list(verification_seeds),
            "gradient_checks": {
                "max_abs_error": max_gradient_error,
                "within_tolerance": max_gradient_error <= tolerance,
                "tolerance": tolerance,
            },
            "symmetric_point": {
                "max_b_block_error": max_b_block_error,
                "max_b_eigenvalue_error": max_b_eigenvalue_error,
                "max_full_max_error": max_full_max_error,
                "within_tolerance": (
                    max_b_block_error <= tolerance
                    and max_b_eigenvalue_error <= tolerance
                    and max_full_max_error <= tolerance
                ),
            },
            "reduced": reduced_summary,
            "step_size": step_size_rows,
            "key_results": {
                "lambda_threshold": 1.0,
                "marginal_curvature_at_lambda_2": marginal_mode_curvature(2.0, spec.num_classes),
                "step_bound_at_lambda_2": discrete_step_bound(2.0, spec.num_classes),
                "max_step_factor_error": max_step_error,
            },
        }
        write_json(summary, run.run_dir / "summary.json")
        event_log.write(
            {
                "event": "complete",
                "runtime_seconds": elapsed,
                "gradient_within_tolerance": summary["gradient_checks"]["within_tolerance"],
                "symmetric_within_tolerance": summary["symmetric_point"]["within_tolerance"],
                "summary_path": "summary.json",
            }
        )

    print(f"Run directory: {run.run_dir.resolve()}")
    print(f"Hessian CSV:   {(run.run_dir / 'hessian_spectrum.csv').resolve()}")
    print(f"Summary:       {(run.run_dir / 'summary.json').resolve()}")
    print(f"Gradient max abs error: {max_gradient_error:.3e}")
    print(f"B-block max abs error:  {max_b_block_error:.3e}")
    print(f"Full-max max abs error: {max_full_max_error:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
