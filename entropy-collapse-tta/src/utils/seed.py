"""Reproducibility utilities: multi-backend random seeding."""

import os
import random
from typing import Optional
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Set random seed across Python, NumPy, PyTorch (CPU and CUDA).

    Args:
        seed: Integer seed value.
        deterministic: If True, configure PyTorch for strict deterministic
                       algorithms (may impact speed slightly, but guarantees reproducibility).

    Returns:
        The seed used.
    """
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
        # If supported by the installed torch version
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass

    return seed
