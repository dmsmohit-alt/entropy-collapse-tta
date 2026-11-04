"""Central random-seed handling.

The project records one integer seed per run.  ``seed_everything`` seeds the
Python, NumPy, and PyTorch generators together so that data order, parameter
initialisation, and any stochastic augmentation are reproducible.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed all RNGs used by the project.

    Args:
        seed: Integer in ``[0, 2**32)``.
        deterministic: If true, request deterministic PyTorch kernels.  This
            can make some GPU operations slower or unavailable; use it for
            final runs and disable it only for profiling.

    ``PYTHONHASHSEED`` only affects child processes, because hash
    randomisation is fixed when the interpreter starts.  It is still set for
    reproducibility of subprocesses launched by future scripts.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must satisfy 0 <= seed < 2**32")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # TF32 changes floating-point reductions on Ampere and newer GPUs.
        # Disabling it trades some throughput for closer run-to-run agreement.
        # Full bitwise reproducibility is still not guaranteed on GPU.
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except AttributeError:  # pragma: no cover - older torch builds
            pass
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except AttributeError:  # pragma: no cover - older torch builds
            pass
        torch.use_deterministic_algorithms(False)


__all__ = ["seed_everything"]
