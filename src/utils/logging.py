"""Structured, append-only logging for metrics and run events.

Metrics are written as one JSON object per line (JSONL).  Compared with a
single in-memory CSV, this keeps completed records usable if a long stream is
interrupted.  Non-finite floats are rejected so that a silent NaN cannot
corrupt an analysis table later.
"""

from __future__ import annotations

import json
import logging as stdlib_logging
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with an explicit timezone."""
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


class JsonlLogger:
    """Append structured records to a JSONL file.

    The handle is flushed after every record by default.  This is intentionally
    slower than buffered CSV writing but makes long adaptation runs recoverable.
    """

    def __init__(self, path: str | Path, *, flush_each_record: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_each_record = flush_each_record
        self._handle: TextIO | None = self.path.open("a", encoding="utf-8", newline="\n")

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def write(self, record: Mapping[str, Any]) -> None:
        """Append one record.  The caller's mapping is not mutated."""
        if self._handle is None:
            raise ValueError("Cannot write to a closed JsonlLogger")
        if not isinstance(record, Mapping):
            raise TypeError(f"record must be a mapping, got {type(record).__name__}")

        payload = dict(record)
        payload.setdefault("timestamp_utc", utc_now())
        # Serialise before writing so a bad value cannot leave a partial line.
        line = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            ensure_ascii=True,
            default=_json_default,
        )
        self._handle.write(line + "\n")
        if self._flush_each_record:
            self._handle.flush()

    def close(self) -> None:
        """Flush and close the file.  Safe to call more than once."""
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of mappings, rejecting malformed lines."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
            records.append(value)
    return records


def configure_console_logging(level: str = "INFO") -> None:
    """Configure a compact console logger for scripts."""
    numeric_level = getattr(stdlib_logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown logging level: {level!r}")
    stdlib_logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stderr,
    )


__all__ = ["JsonlLogger", "configure_console_logging", "read_jsonl", "utc_now"]
