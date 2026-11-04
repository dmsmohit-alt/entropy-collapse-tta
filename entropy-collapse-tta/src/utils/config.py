"""Configuration management with YAML parsing and serialization."""

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


def load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        path: Path to YAML config.

    Returns:
        Dictionary of configuration options.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def save_yaml_config(config: Union[Dict[str, Any], Any], path: Union[str, Path]) -> Path:
    """Save configuration to YAML file for run provenance.

    Args:
        config: Dict or dataclass config.
        path: Destination file path.

    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if is_dataclass(config):
        data = asdict(config)
    elif isinstance(config, dict):
        data = config
    else:
        raise TypeError(f"Config must be dict or dataclass, got {type(config)}")

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    return path


def merge_configs(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merge override dictionary into base dictionary.

    Args:
        base: Base configuration dict.
        override: Overriding configuration dict.

    Returns:
        Merged configuration dictionary.
    """
    merged = dict(base)
    if override is None:
        return merged

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged
