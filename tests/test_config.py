"""Tests for YAML config loading, merging, hashing, and validation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from utils.config import (
    ConfigError,
    apply_overrides,
    config_hash,
    deep_merge,
    load_config,
    load_yaml,
    save_config,
    validate_run_config,
)

BASE_YAML = """\
version: 1
experiment: base
seed: 0
device: cpu
output_dir: results
logging:
  level: INFO
  jsonl: true
train:
  batch_size: 16
  lr: 0.001
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_base_inheritance_and_dotted_overrides(tmp_path: Path) -> None:
    base = _write(tmp_path / "base.yaml", BASE_YAML)
    child = _write(
        tmp_path / "child.yaml",
        "_base_: base.yaml\nexperiment: child\ntrain:\n  batch_size: 32\n",
    )

    config = load_config(child, overrides=["seed=7", "logging.jsonl=false"])

    assert config["experiment"] == "child"
    assert config["seed"] == 7
    assert config["train"]["batch_size"] == 32
    assert config["train"]["lr"] == pytest.approx(0.001)
    assert config["logging"]["jsonl"] is False
    # The child override must not mutate the base file.
    assert load_yaml(base)["train"]["batch_size"] == 16


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"model": {"width": 8, "depth": 2}, "seed": 1}
    override = {"model": {"width": 16}, "seed": 2}
    base_before = copy.deepcopy(base)
    override_before = copy.deepcopy(override)

    merged = deep_merge(base, override)

    assert merged == {"model": {"width": 16, "depth": 2}, "seed": 2}
    assert base == base_before
    assert override == override_before


def test_overrides_parse_yaml_scalars_and_create_nested_mappings() -> None:
    config = {"train": {"batch_size": 16}, "logging": {"jsonl": True}}

    updated = apply_overrides(
        config,
        ["train.batch_size=64", "logging.jsonl=false", "new_section.null_value=null"],
    )

    assert updated["train"]["batch_size"] == 64
    assert updated["logging"]["jsonl"] is False
    assert updated["new_section"]["null_value"] is None
    assert config["train"]["batch_size"] == 16


def test_override_cannot_descend_into_scalar() -> None:
    with pytest.raises(ConfigError, match="not a mapping"):
        apply_overrides({"seed": 0}, ["seed.value=1"])


def test_missing_file_invalid_root_and_inheritance_cycle(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")

    invalid = _write(tmp_path / "list.yaml", "- a\n- b\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_yaml(invalid)

    _write(tmp_path / "a.yaml", "_base_: b.yaml\n")
    _write(tmp_path / "b.yaml", "_base_: a.yaml\n")
    with pytest.raises(ConfigError, match="cycle"):
        load_config(tmp_path / "a.yaml")


def test_config_hash_is_stable_under_key_order_and_sensitive_to_values() -> None:
    first = {"a": 1, "b": {"x": 2, "y": 3}}
    reordered = {"b": {"y": 3, "x": 2}, "a": 1}
    changed = {"a": 1, "b": {"x": 2, "y": 4}}

    assert config_hash(first) == config_hash(reordered)
    assert config_hash(first) != config_hash(changed)


def test_save_config_round_trips(tmp_path: Path) -> None:
    config = load_yaml(_write(tmp_path / "base.yaml", BASE_YAML))
    destination = save_config(config, tmp_path / "nested" / "resolved.yaml")

    assert load_yaml(destination) == config


def test_validate_run_config_accepts_valid_config(tmp_path: Path) -> None:
    validate_run_config(load_yaml(_write(tmp_path / "valid.yaml", BASE_YAML)))


def test_validate_run_config_rejects_bad_core_fields() -> None:
    valid = {
        "version": 1,
        "experiment": "unit_test",
        "seed": 0,
        "device": "cpu",
        "output_dir": "results",
        "logging": {"level": "INFO", "jsonl": True},
    }

    for key in ("experiment", "seed", "device", "logging"):
        missing = dict(valid)
        del missing[key]
        with pytest.raises(ConfigError, match="Missing"):
            validate_run_config(missing)

    with pytest.raises(ConfigError, match="path separators"):
        validate_run_config({**valid, "experiment": "bad/name"})
    with pytest.raises(ConfigError, match="seed"):
        validate_run_config({**valid, "seed": True})
    with pytest.raises(ConfigError, match="device"):
        validate_run_config({**valid, "device": "gpu"})
    with pytest.raises(ConfigError, match="logging.jsonl"):
        validate_run_config({**valid, "logging": {"level": "INFO", "jsonl": "yes"}})
