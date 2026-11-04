"""Typed helpers for loading, merging, validating, and hashing YAML configs.

A child config can inherit from a base file with

    _base_: base.yaml

Paths are resolved relative to the child config.  Command-line style
overrides such as ``train.batch_size=32`` are applied after inheritance,
using YAML scalar parsing for the value.

Every run saves the fully resolved config, not just the child file.  The
config hash is stable under mapping-key order, which makes run directories
and logs easy to audit.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import yaml

_BASE_KEY = "_base_"
_DEVICE_PATTERN = re.compile(r"^(auto|cpu|cuda(:\d+)?|mps)$")


class ConfigError(ValueError):
    """Raised when a config file or override is invalid."""


def _require_mapping(value: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{source} must contain a YAML mapping, got {type(value).__name__}")
    return dict(value)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``."""
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if raw is None:
        return {}
    return _require_mapping(raw, source=str(config_path))


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_with_base(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved in stack:
        cycle = " -> ".join(str(item) for item in (*stack, resolved))
        raise ConfigError(f"Config inheritance cycle: {cycle}")
    if not resolved.is_file():
        raise ConfigError(f"Config file not found: {resolved}")

    raw = load_yaml(resolved)
    base_ref = raw.pop(_BASE_KEY, None)
    if base_ref is None:
        return raw
    if not isinstance(base_ref, str) or not base_ref.strip():
        raise ConfigError(f"{_BASE_KEY} in {resolved} must be a non-empty path string")

    base_path = (resolved.parent / base_ref).resolve()
    base = _load_with_base(base_path, (*stack, resolved))
    return deep_merge(base, raw)


def _parse_override(override: str) -> tuple[tuple[str, ...], Any]:
    if "=" not in override:
        raise ConfigError(f"Override must look like key=value, got {override!r}")
    dotted_key, raw_value = override.split("=", 1)
    parts = tuple(part.strip() for part in dotted_key.split("."))
    if not parts or any(not part for part in parts):
        raise ConfigError(f"Override has an empty key component: {override!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML value in override {override!r}: {exc}") from exc
    return parts, value


def apply_overrides(
    config: Mapping[str, Any],
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Apply dotted ``key=value`` overrides to a deep copy of ``config``."""
    result = copy.deepcopy(dict(config))
    for override in overrides:
        parts, value = _parse_override(override)
        cursor: dict[str, Any] = result
        for part in parts[:-1]:
            existing = cursor.get(part)
            if existing is None:
                existing = {}
                cursor[part] = existing
            if not isinstance(existing, Mapping):
                raise ConfigError(
                    f"Cannot set nested override {override!r}: {part!r} is not a mapping"
                )
            # Copy in case a Mapping implementation is read-only or shared.
            child = dict(existing)
            cursor[part] = child
            cursor = child
        cursor[parts[-1]] = value
    return result


def load_config(
    path: str | Path,
    *,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Load a config, resolve its base chain, and apply overrides."""
    return apply_overrides(_load_with_base(Path(path), ()), overrides)


def config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash of a JSON-serialisable config mapping."""
    canonical = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Write a resolved config to YAML and return its path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(
            dict(config),
            handle,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    return destination


def validate_run_config(config: Mapping[str, Any]) -> None:
    """Validate the fields shared by every experiment config.

    Experiment-specific sections are validated by their own modules later.
    """
    required = ("version", "experiment", "seed", "device", "output_dir", "logging")
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"Missing required config keys: {', '.join(missing)}")

    if isinstance(config["version"], bool) or not isinstance(config["version"], int):
        raise ConfigError("version must be an integer")
    if config["version"] < 1:
        raise ConfigError("version must be >= 1")

    experiment = config["experiment"]
    if not isinstance(experiment, str) or not experiment.strip():
        raise ConfigError("experiment must be a non-empty string")
    if Path(experiment).name != experiment or experiment in {".", ".."}:
        raise ConfigError("experiment must not contain path separators")

    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError("seed must be an integer")
    if not 0 <= seed < 2**32:
        raise ConfigError("seed must satisfy 0 <= seed < 2**32 for NumPy compatibility")

    device = config["device"]
    if not isinstance(device, str) or not _DEVICE_PATTERN.fullmatch(device):
        raise ConfigError("device must be one of auto, cpu, mps, cuda, or cuda:<index>")

    output_dir = config["output_dir"]
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ConfigError("output_dir must be a non-empty path string")

    logging_config = config["logging"]
    if not isinstance(logging_config, Mapping):
        raise ConfigError("logging must be a mapping")
    if not isinstance(logging_config.get("level"), str):
        raise ConfigError("logging.level must be a string")
    if not isinstance(logging_config.get("jsonl"), bool):
        raise ConfigError("logging.jsonl must be a boolean")


__all__ = [
    "ConfigError",
    "apply_overrides",
    "config_hash",
    "deep_merge",
    "load_config",
    "load_yaml",
    "save_config",
    "validate_run_config",
]
