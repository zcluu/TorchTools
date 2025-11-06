import logging
import os
import random
from typing import Optional, Union

import numpy as np

import torch

_log: logging.Logger = logging.getLogger(__name__)


def seed(seed: int, deterministic: Optional[Union[str, int]] = None) -> None:
    max_val = np.iinfo(np.uint32).max
    min_val = np.iinfo(np.uint32).min
    if seed < min_val or seed > max_val:
        raise ValueError(
            f"Invalid seed value provided: {seed}. Value must be in the range [{min_val}, {max_val}]"
        )
    _log.debug(f"Setting seed to {seed}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic is not None:
        _log.debug(f"Setting deterministic debug mode to {deterministic}")
        torch.set_deterministic_debug_mode(deterministic)
        deterministic_debug_mode = torch.get_deterministic_debug_mode()
        if deterministic_debug_mode == 0:
            _log.debug("Disabling cuDNN deterministic mode")
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            _log.debug("Enabling cuDNN deterministic mode")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # reference: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
