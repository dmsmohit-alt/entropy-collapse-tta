"""Tests for append-only JSONL logging."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from utils.logging import JsonlLogger, read_jsonl


def test_jsonl_logger_flushes_and_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"

    with JsonlLogger(path) as logger:
        logger.write({"event": "start", "step": 0, "path": tmp_path})
        logger.write({"event": "metric", "step": 1, "accuracy": 0.5})
        # Read before closing to verify that flush_each_record works.
        records = read_jsonl(path)

    assert len(records) == 2
    assert records[0]["event"] == "start"
    assert records[0]["path"] == str(tmp_path)
    assert "timestamp_utc" in records[0]
    assert records[1]["accuracy"] == pytest.approx(0.5)


def test_jsonl_logger_rejects_nan_without_writing_partial_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    logger = JsonlLogger(path)

    with pytest.raises(ValueError, match="Out of range float values"):
        logger.write({"step": 0, "loss": math.nan})
    logger.close()

    assert path.read_text(encoding="utf-8") == ""


def test_jsonl_logger_rejects_non_mapping_and_writes_after_close(tmp_path: Path) -> None:
    logger = JsonlLogger(tmp_path / "metrics.jsonl")
    with pytest.raises(TypeError, match="mapping"):
        logger.write([("step", 0)])  # type: ignore[arg-type]
    logger.close()

    with pytest.raises(ValueError, match="closed"):
        logger.write({"step": 1})
