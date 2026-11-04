"""Phase 1: entropy-minimisation dynamics and collapse-basin scan.

The script runs a pre-registered (in YAML) bias grid over multiple data seeds,
saves every per-run metric, and writes aggregate tables.  It does not fit or
claim a theoretical threshold; Phase 2 handles the derivation.
"""

from __future__ import annotations

import argparse
import json
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

from theory.basin import (  # noqa: E402
    make_bias_grid,
    run_basin_scan,
    summarize_basin,
    validate_seeds,
    write_json,
    write_scan_csv,
    write_summary_csv,
)
from theory.toy_model import (  # noqa: E402
    ToySetup,
    make_gaussian_mixture,
    simulate_entropy_minimisation,
)
from utils.config import config_hash, load_config, validate_run_config  # noqa: E402
from utils.logging import JsonlLogger, configure_console_logging, utc_now  # noqa: E402
from utils.run import create_run_dir  # noqa: E402
from utils.seeding import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "theory_phase1.yaml",
        help="Phase 1 YAML config.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override; may be repeated.",
    )
    return parser.parse_args()


def _reasonably_positive_float(section: dict[str, Any], key: str) -> float:
    value = float(section[key])
    if value <= 0:
        raise ValueError(f"{key} must be > 0")
    return value


def _trajectory_name(seed: int, bias: float) -> str:
    return f"seed{seed}_bias{bias:.2f}.jsonl"


def _write_trajectory(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    with JsonlLogger(path) as logger:
        for record in records:
            logger.write(record)


def main() -> int:
    args = parse_args()
    config = load_config(args.config, overrides=args.override)
    validate_run_config(config)
    configure_console_logging(str(config["logging"]["level"]))
    seed_everything(int(config["seed"]), deterministic=True)

    if "toy" not in config or not isinstance(config["toy"], dict):
        raise ValueError("Config must contain a toy section")
    setup = ToySetup.from_mapping(config["toy"])

    basin_config = config["basin"]
    seeds = validate_seeds(basin_config["seeds"])
    biases = make_bias_grid(
        float(basin_config["start"]),
        float(basin_config["stop"]),
        _reasonably_positive_float(basin_config, "step"),
    )

    trajectory_config = config["trajectory"]
    trajectory_biases = tuple(float(value) for value in trajectory_config["biases"])
    trajectory_seeds = validate_seeds(trajectory_config.get("seeds", seeds))
    if any(bias < 0 for bias in trajectory_biases):
        raise ValueError("trajectory biases must be >= 0")

    run = create_run_dir(config)
    start_time = time.perf_counter()
    (run.run_dir / "trajectories").mkdir(parents=True, exist_ok=True)

    with JsonlLogger(run.run_dir / "run_events.jsonl") as event_log:
        event_log.write(
            {
                "event": "start",
                "config_hash": config_hash(config),
                "seeds": list(seeds),
                "bias_grid_count": len(biases),
                "trajectory_biases": list(trajectory_biases),
                "trajectory_seeds": list(trajectory_seeds),
            }
        )

        print(f"Scanning {len(seeds)} seeds x {len(biases)} bias values...")
        scan_rows, source_rows = run_basin_scan(setup, seeds=seeds, biases=biases)
        write_scan_csv(scan_rows, run.run_dir / "basin_scan.csv")
        write_json({"source_rows": source_rows}, run.run_dir / "source_metrics.json")

        summary = summarize_basin(
            scan_rows,
            source_rows=source_rows,
            setup=setup,
            seeds=seeds,
            biases=biases,
        )

        trajectory_rows: list[dict[str, Any]] = []
        for seed in trajectory_seeds:
            data = make_gaussian_mixture(setup, seed=seed)
            for bias in trajectory_biases:
                result = simulate_entropy_minimisation(data, setup, initial_bias=bias)
                trajectory_path = run.run_dir / "trajectories" / _trajectory_name(seed, bias)
                _write_trajectory(trajectory_path, result.records)
                trajectory_rows.append(
                    {
                        "seed": seed,
                        "initial_bias": bias,
                        "collapse_step": result.collapse_step,
                        "collapse_onset_step": result.onset_step,
                        "final_marginal_entropy": result.final["eval_marginal_entropy"],
                        "final_top_class_fraction": result.final["eval_top_class_fraction"],
                        "final_accuracy": result.final["eval_accuracy"],
                        "path": str(trajectory_path.relative_to(run.run_dir)),
                    }
                )

        elapsed = time.perf_counter() - start_time
        summary.update(
            {
                "experiment": config["experiment"],
                "created_utc": utc_now(),
                "runtime_seconds": elapsed,
                "toy_setup": asdict(setup),
                "trajectories": trajectory_rows,
            }
        )
        write_json(summary, run.run_dir / "summary.json")
        write_summary_csv(summary, run.run_dir / "basin_summary.csv")
        event_log.write(
            {
                "event": "complete",
                "runtime_seconds": elapsed,
                "threshold": summary["threshold"],
                "summary_path": "summary.json",
            }
        )

    print(f"Run directory: {run.run_dir.resolve()}")
    print(f"Basin CSV:     {(run.run_dir / 'basin_scan.csv').resolve()}")
    print(f"Summary:       {(run.run_dir / 'summary.json').resolve()}")
    print(json.dumps(summary["threshold"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
