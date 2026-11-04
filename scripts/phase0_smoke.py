"""End-to-end Phase 0 wiring check.

This script is deliberately *not* a scientific experiment.  It loads a YAML
config, creates a seeded run directory, computes the entropy metrics on a
small synthetic batch, and writes the result to JSONL.  Its purpose is to
verify that configuration, seeding, run metadata, metrics, and logging all
work together before the toy-model experiments begin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402

from metrics.entropy import (  # noqa: E402
    entropy_gap,
    marginal_entropy,
    predictive_entropy,
    top_class_fraction,
)
from utils.config import config_hash, load_config, validate_run_config  # noqa: E402
from utils.logging import JsonlLogger, configure_console_logging  # noqa: E402
from utils.run import create_run_dir  # noqa: E402
from utils.seeding import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "phase0_smoke.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override; may be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, overrides=args.override)
    validate_run_config(config)
    configure_console_logging(str(config["logging"]["level"]))

    seed = int(config["seed"])
    seed_everything(seed, deterministic=True)

    run = create_run_dir(config)

    # A small batch with a deliberate class bias, only to exercise the metrics.
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(16, 3, generator=generator) * 1.25
    logits[:, 0] += 0.75

    record = {
        "event": "phase0_smoke",
        "step": 0,
        "seed": seed,
        "config_hash": config_hash(config),
        "num_samples": int(logits.shape[0]),
        "num_classes": int(logits.shape[1]),
        "mean_predictive_entropy": float(predictive_entropy(logits)),
        "marginal_entropy": float(marginal_entropy(logits)),
        "entropy_gap": float(entropy_gap(logits)),
        "top_class_fraction": float(top_class_fraction(logits)),
        "note": "synthetic wiring check; not a scientific result",
    }

    with JsonlLogger(run.metrics_path) as logger:
        logger.write(record)

    print(f"Phase 0 smoke run directory: {run.run_dir.resolve()}")
    print(f"Resolved config:            {run.config_path.resolve()}")
    print(f"Metrics:                    {run.metrics_path.resolve()}")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
