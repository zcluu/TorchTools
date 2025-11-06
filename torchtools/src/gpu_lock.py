import os
import fcntl
import time
import warnings
from contextlib import contextmanager
from typing import Optional

import torch

LOCK_DIR = os.environ.get("TORCHTOOLS_LOCK_DIR", "/tmp")
LOCK_PREFIX = "torchtools_gpu_lock"


def _lock_file_path(gpu_index: int) -> str:
    return os.path.join(LOCK_DIR, f"{LOCK_PREFIX}_{gpu_index}.lock")


@contextmanager
def reserve_gpu_for_training(
    gpu: Optional[int] = None,
    training_reserved_bytes: int = 0,
    safety_margin_bytes: int = 50 * 1024 * 1024,
    blocking: bool = True,
    timeout: Optional[float] = None,
    fence: bool = True,
):
    """
    Context manager that cooperatively reserves a CUDA GPU for a training run.

    Strategy:
    - Acquire a filesystem lock on /tmp/torchtools_gpu_lock_{gpu} so other cooperative
      processes (that use this same helper) will avoid this GPU.
    - Optionally allocate a "fence" tensor on the GPU occupying the currently
      available free memory minus the amount you will need for your training
      (``training_reserved_bytes``) and a safety margin. The fence prevents other
      processes from grabbing the leftover free memory while your training runs.

    Notes / assumptions:
    - This is a cooperative approach. It will not stop arbitrary processes that
      do not use this helper. For stronger enforcement you need privileged
      operations (not provided here).
    - The caller should run the training code in the same process (or at least
      ensure the training allocates after entering this context) so memory
      accounting stays stable. The helper will try to leave ``training_reserved_bytes``
      available for training.

    Parameters:
    - gpu: GPU index (int). If None, uses current CUDA device.
    - training_reserved_bytes: number of bytes expected to be used by training.
    - safety_margin_bytes: extra bytes to leave free as a safety margin.
    - blocking: whether to block when acquiring the filesystem lock.
    - timeout: optional timeout (seconds) when blocking is True.
    - fence: whether to attempt allocating the fence tensor. If False, only file lock is used.

    Example:
        with reserve_gpu_for_training(gpu=0, training_reserved_bytes=6*1024**3):
            # build model / run training here

    """
    # If CUDA not available, nothing to do
    if not torch.cuda.is_available():
        yield
        return

    # determine GPU index
    if gpu is None:
        gpu_index = torch.cuda.current_device()
    elif isinstance(gpu, int):
        gpu_index = gpu
    else:
        raise TypeError("gpu must be an int index or None")

    lock_path = _lock_file_path(gpu_index)
    fd = open(lock_path, "w")

    # acquire file lock
    start = time.time()
    while True:
        try:
            flags = fcntl.LOCK_EX
            if not blocking:
                flags |= fcntl.LOCK_NB
            fcntl.flock(fd.fileno(), flags)
            break
        except BlockingIOError:
            if not blocking:
                fd.close()
                raise RuntimeError(f"GPU {gpu_index} is locked by another process")
            if timeout is not None and (time.time() - start) >= timeout:
                fd.close()
                raise TimeoutError(f"Timeout while waiting for GPU {gpu_index} lock")
            time.sleep(0.2)

    fence_tensor = None
    try:
        if fence:
            # try to allocate a fence that occupies free memory except the training part
            try:
                torch.cuda.set_device(gpu_index)
            except Exception:
                # if setting device fails, continue with file lock only
                warnings.warn(
                    "Could not set CUDA device for fence allocation; proceeding with file lock only"
                )
                yield
                return

            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_index)
            except Exception:
                # older torch maybe different API; fall back to 0
                free_bytes, total_bytes = 0, 0

            target = int(
                free_bytes - int(training_reserved_bytes) - int(safety_margin_bytes)
            )
            min_alloc = 10 * 1024 * 1024  # 10 MB minimum to bother allocating

            if target < min_alloc:
                # Nothing sensible to allocate; proceed with lock only
                warnings.warn(
                    "Not enough free GPU memory to allocate a fence tensor; proceeding with file lock only"
                )
            else:
                # Try allocating; if it fails, reduce by halves until successful or too small
                size = target
                allocated = None
                while size >= min_alloc:
                    try:
                        # allocate as uint8 so number of elements == bytes
                        fence_tensor = torch.empty(
                            size, dtype=torch.uint8, device=f"cuda:{gpu_index}"
                        )
                        # Touch the tensor to ensure allocation (some backends lazy-allocate)
                        fence_tensor.fill_(0)
                        allocated = fence_tensor
                        break
                    except RuntimeError:
                        # reduce target and retry
                        size = size // 2

                if fence_tensor is None:
                    warnings.warn(
                        "Failed to allocate fence tensor; proceeding with file lock only"
                    )

        # yield control to caller while lock (+ optional fence) held
        yield

    finally:
        # release fence tensor first
        try:
            if fence_tensor is not None:
                # free GPU memory deterministically
                del fence_tensor
                # run a small cuda empty cache to encourage free (no-op if not available)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass

        # release file lock and close fd
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            fd.close()
        except Exception:
            pass
