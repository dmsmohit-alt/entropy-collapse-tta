"""Unit tests for utility functions (seeding, logging, config)."""

import json
from pathlib import Path
import tempfile
import torch
import numpy as np

from src.utils.seed import seed_everything
from src.utils.config import load_yaml_config, save_yaml_config, merge_configs
from src.utils.logging import MetricLogger, setup_logger


def test_seed_everything_reproducibility() -> None:
    """Check that seed_everything produces deterministic tensors and arrays."""
    seed_everything(1337)
    t1 = torch.randn(10)
    n1 = np.random.randn(10)

    seed_everything(1337)
    t2 = torch.randn(10)
    n2 = np.random.randn(10)

    assert torch.equal(t1, t2)
    assert np.array_equal(n1, n2)


def test_yaml_config_io_and_merge() -> None:
    """Test saving, loading, and merging YAML configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "test_cfg.yaml"
        original = {
            "experiment": {"seed": 42, "name": "base"},
            "lr": 0.01,
        }
        save_yaml_config(original, cfg_path)
        loaded = load_yaml_config(cfg_path)
        assert loaded == original

        override = {"experiment": {"seed": 99}, "new_key": True}
        merged = merge_configs(loaded, override)
        assert merged["experiment"]["seed"] == 99
        assert merged["experiment"]["name"] == "base"
        assert merged["new_key"] is True


def test_metric_logger_records_and_exports() -> None:
    """Verify that MetricLogger accurately saves per-step CSV and summary JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricLogger(output_dir=tmpdir, experiment_name="test_exp")
        for step in range(5):
            logger.log_step(step, {"loss": 1.0 / (step + 1), "acc": 0.2 * step})

        csv_file = logger.save_csv()
        assert csv_file.exists()

        summary_file = logger.save_summary_json({"final_acc": 0.8, "status": "success"})
        assert summary_file.exists()

        with open(summary_file, "r") as f:
            data = json.load(f)
            assert data["final_acc"] == 0.8
