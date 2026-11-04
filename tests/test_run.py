"""Tests for run-directory creation and reproducibility metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from utils.config import config_hash, load_yaml
from utils.run import create_run_dir


def _config(output_dir: Path) -> dict[str, object]:
    return {
        "version": 1,
        "experiment": "unit_test",
        "seed": 3,
        "seeds": [3, 4],
        "batch_size": 64,
        "tolerance": 1.0e-4,
        "device": "cpu",
        "output_dir": str(output_dir),
        "logging": {"level": "INFO", "jsonl": True},
    }


def test_create_run_dir_saves_resolved_config_and_metadata(tmp_path: Path) -> None:
    config = _config(tmp_path / "results")
    fixed_time = datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)

    info = create_run_dir(config, timestamp=fixed_time)

    assert info.run_dir.is_dir()
    assert info.run_id.startswith("20260102T030405.123456Z_unit_test_seed3_")
    assert load_yaml(info.config_path) == config
    assert info.metrics_path == info.run_dir / "metrics.jsonl"

    metadata = json.loads(info.metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_id"] == info.run_id
    assert metadata["seed"] == 3
    assert metadata["seeds"] == [3, 4]
    assert metadata["batch_size"] == 64
    assert metadata["tolerance"] == 1.0e-4
    assert metadata["config_hash"] == config_hash(config)
    assert metadata["created_utc"] == fixed_time.isoformat()
    assert "python_version" in metadata
    assert "platform" in metadata

    runtime = metadata["runtime"]
    assert runtime["requested_device"] == "cpu"
    assert runtime["resolved_device"] == "cpu"
    assert runtime["device"]["name"] == "cpu"
    assert "torch_version" in runtime
    assert "torch_package_version" in runtime
    assert "cuda_version" in runtime
    assert "version" in runtime["cudnn"]
    assert "deterministic" in runtime["cudnn"]
    assert "reproducibility_note" in metadata


def test_run_dirs_are_distinct_when_timestamps_differ(tmp_path: Path) -> None:
    config = _config(tmp_path / "results")
    first = create_run_dir(
        config, timestamp=datetime(2026, 1, 2, 3, 4, 5, 0, tzinfo=timezone.utc)
    )
    second = create_run_dir(
        config, timestamp=datetime(2026, 1, 2, 3, 4, 6, 0, tzinfo=timezone.utc)
    )

    assert first.run_dir != second.run_dir
    assert first.config_path.exists()
    assert second.config_path.exists()
