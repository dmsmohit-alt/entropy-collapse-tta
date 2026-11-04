"""Check that the active interpreter matches ``requirements.lock.txt``.

This is a guard against the kind of silent drift that produced a torch
package-metadata version (2.15.0) different from the runtime version
(2.9.1+cu126) in earlier saved runs.  It compares every pinned distribution
and prints the Python, CUDA, cuDNN, and GPU details used for Phase 3.
"""

from __future__ import annotations

import argparse
import platform
import re
import sys
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
DEFAULT_LOCK = PROJECT_ROOT / "requirements.lock.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_LOCK,
        help="Path to the lock file.",
    )
    return parser.parse_args()


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def read_lock(path: Path) -> dict[str, str]:
    """Return ``{normalized_name: exact_version}`` from a pip-style lock file."""
    requirements: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise ValueError(f"{path}:{line_number}: expected name==version, got {raw!r}")
        name, version = line.split("==", 1)
        requirements[_normalize(name)] = version.strip()
    return requirements


def main() -> int:
    args = parse_args()
    requirements = read_lock(args.lock)

    missing: list[str] = []
    mismatched: list[tuple[str, str, str]] = []
    for name, expected in requirements.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if actual != expected:
            mismatched.append((name, expected, actual))

    print(f"Python: {platform.python_version()} ({platform.platform()})")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency check
        print(f"torch import failed: {exc!r}")
        return 1

    print(f"torch:  {torch.__version__} (package metadata {metadata.version('torch')})")
    print(f"CUDA:   {torch.version.cuda}; available={torch.cuda.is_available()}")
    print(f"cuDNN:  {torch.backends.cudnn.version()}; enabled={torch.backends.cudnn.enabled}")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        print(
            f"GPU:    {properties.name}, "
            f"{properties.total_memory / 1024**3:.1f} GiB, "
            f"capability {properties.major}.{properties.minor}"
        )

    # Show the flags that saved runs will actually use, not the process defaults.
    from utils.seeding import seed_everything

    seed_everything(0, deterministic=True)
    print(
        "determinism (after seed_everything): "
        f"cudnn.deterministic={torch.backends.cudnn.deterministic}, "
        f"cudnn.benchmark={torch.backends.cudnn.benchmark}, "
        f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}, "
        f"use_deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}"
    )

    if missing or mismatched:
        for name in missing:
            print(f"MISSING  {name} (expected {requirements[name]})")
        for name, expected, actual in mismatched:
            print(f"MISMATCH {name}: expected {expected}, found {actual}")
        return 1

    print(f"Lock OK: {len(requirements)} pinned distributions match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
