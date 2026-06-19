"""One place to make a notebook reproducible. Call `set_seed(SEED)` once near
the top, after the config cell."""

from __future__ import annotations

import os
import random


def set_seed(seed: int = 0, deterministic: bool = True) -> int:
    """Seed Python, NumPy and (if present) torch. Returns the seed so you can
    print it in a checkpoint. `deterministic=True` also pins cuDNN so two runs
    on the same GPU match — slightly slower, but worth it for a class."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed
