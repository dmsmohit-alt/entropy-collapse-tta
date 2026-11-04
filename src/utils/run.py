"""Run directory creation and reproducibility metadata.

Each run gets an immutable directory containing the fully resolved config and
a metadata file.  Metrics are appended separately by ``JsonlLogger``.  The
directory name includes a config hash, so two runs with different settings do
not silently share a location.
"""

from __future__ import annotations

import json
import platform
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from utils.config import config_hash, save_config, validate_run_config


@dataclass(frozen=True)
class RunInfo:
    """Paths and identifiers for one reproducible run."""

    run_id: str
    run_dir: Path
    config_path: Path
    metadata_path: Path
    metrics_path: Path


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _resolve_requested_device(requested: str) -> str:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if requested != "auto":
        return requested
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a declared dependency
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _torch_runtime(requested_device: str) -> dict[str, Any]:
    """Capture versions, cuDNN flags, and GPU identity for run metadata.

    ``torch.__version__`` and the package metadata version are recorded
    separately.  A transient mismatch between the two was observed earlier in
    this project, so a reproducibility check must not rely on only one of them.
    The function never raises when CUDA is unavailable; it records the CPU
    fallback instead.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a declared dependency
        return {"available": False, "error": repr(exc)}

    resolved = _resolve_requested_device(str(requested_device))
    runtime: dict[str, Any] = {
        "requested_device": str(requested_device),
        "resolved_device": resolved,
        "torch_version": torch.__version__,
        "torch_package_version": _package_version("torch"),
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn": {
            "version": torch.backends.cudnn.version(),
            "enabled": bool(torch.backends.cudnn.enabled),
            "deterministic": bool(torch.backends.cudnn.deterministic),
            "benchmark": bool(torch.backends.cudnn.benchmark),
            "allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "device": {"name": "cpu"},
    }
    try:
        runtime["matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:  # pragma: no cover - depends on the torch build
        runtime["matmul_allow_tf32"] = None

    if runtime["cuda_available"] and resolved.startswith("cuda"):
        index = 0
        if ":" in resolved:
            index = int(resolved.split(":", 1)[1])
        properties = torch.cuda.get_device_properties(index)
        runtime["device"] = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "total_memory_gib": round(properties.total_memory / 1024**3, 3),
            "capability": f"{properties.major}.{properties.minor}",
            "multi_processor_count": int(properties.multi_processor_count),
        }
    return runtime


def _safe_experiment_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe or "experiment"


def create_run_dir(
    config: Mapping[str, Any],
    *,
    timestamp: datetime | None = None,
) -> RunInfo:
    """Create a unique run directory and persist config plus environment data.

    ``config["output_dir"]`` is the results root.  A timestamp with
    microseconds avoids collisions when several short runs start together.
    """
    validate_run_config(config)

    now = timestamp or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    seed = int(config["seed"])
    experiment = str(config["experiment"])
    safe_experiment = _safe_experiment_name(experiment)
    digest = config_hash(config)
    stamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = f"{stamp}_{safe_experiment}_seed{seed}_{digest[:8]}"

    results_root = Path(config["output_dir"]).expanduser()
    run_dir = results_root / safe_experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    config_path = save_config(config, run_dir / "config.yaml")
    metadata_path = run_dir / "run_metadata.json"
    seeds = config.get("seeds", [seed])
    metadata_payload = {
        "run_id": run_id,
        "experiment": experiment,
        "seed": seed,
        "seeds": seeds,
        "batch_size": config.get("batch_size"),
        "tolerance": config.get("tolerance"),
        "config_hash": digest,
        "created_utc": now.isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": _package_version("numpy"),
            "torch": _package_version("torch"),
            "PyYAML": _package_version("PyYAML"),
        },
        "runtime": _torch_runtime(str(config.get("device", "auto"))),
        "reproducibility_note": (
            "GPU reductions are not guaranteed to be bitwise reproducible. "
            "Compare repeated runs using the configured tolerance."
        ),
        "argv": sys.argv,
    }
    with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata_payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    return RunInfo(
        run_id=run_id,
        run_dir=run_dir,
        config_path=config_path,
        metadata_path=metadata_path,
        metrics_path=run_dir / "metrics.jsonl",
    )


__all__ = ["RunInfo", "create_run_dir"]
