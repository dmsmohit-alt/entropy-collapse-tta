"""Structured logging and experiment trajectory recording."""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def setup_logger(
    name: str = "entropy_collapse",
    log_dir: Optional[str] = None,
    filename: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a multi-stream logger (console + optional file).

    Args:
        name: Logger name.
        log_dir: Directory where log file should be saved.
        filename: Name of log file (e.g. 'run_0.log').
        level: Logging verbosity level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logger called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir is not None and filename is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, filename)
        file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricLogger:
    """Records per-step metrics and exports trajectories to CSV and JSON.

    Ensures no metric values are lost and guarantees reproducible post-hoc plotting.
    """

    def __init__(self, output_dir: str, experiment_name: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.records: List[Dict[str, Any]] = []

    def log_step(self, step: int, metrics: Dict[str, Any]) -> None:
        """Record metrics for a single step.

        Args:
            step: Current step index.
            metrics: Key-value metric mapping (e.g. {'loss': 0.1, 'acc': 0.85}).
        """
        row = {"step": step}
        for k, v in metrics.items():
            if hasattr(v, "item"):
                row[k] = v.item()
            else:
                row[k] = v
        self.records.append(row)

    def save_csv(self, filename: Optional[str] = None) -> Path:
        """Export all logged steps to CSV.

        Returns:
            Path to saved CSV file.
        """
        fname = filename or f"{self.experiment_name}_metrics.csv"
        csv_path = self.output_dir / fname
        if not self.records:
            return csv_path

        fieldnames = list(self.records[0].keys())
        # Guarantee all keys present across records are included
        all_keys = set()
        for r in self.records:
            all_keys.update(r.keys())
        fieldnames = sorted(list(all_keys), key=lambda x: (x != "step", x))

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow(r)

        return csv_path

    def save_summary_json(self, summary_data: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """Save high-level summary metrics (final acc, collapse step, etc.) to JSON."""
        fname = filename or f"{self.experiment_name}_summary.json"
        json_path = self.output_dir / fname
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        return json_path
