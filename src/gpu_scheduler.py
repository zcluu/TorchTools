import time
from typing import Any, Callable, Optional, Tuple
import multiprocessing as mp

import torch

from .gpu_lock import reserve_gpu_for_training


def _probe_worker(
    fn: Callable, args: Tuple, kwargs: dict, q: mp.Queue, gpu: Optional[int]
):
    """Worker run in a separate process to probe peak GPU memory usage."""
    try:
        if gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(gpu)
            # reset stats
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        # run probe function
        fn(*args, **kwargs)

        if torch.cuda.is_available():
            # synchronize and get peak
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                peak = torch.cuda.max_memory_allocated()
            except Exception:
                peak = 0
        else:
            peak = 0

        q.put((True, int(peak)))
    except Exception as e:
        q.put((False, str(e)))


def estimate_gpu_usage_by_probe(
    probe_fn: Callable,
    args: Tuple = (),
    kwargs: dict = None,
    gpu: Optional[int] = None,
    timeout: float = 60.0,
) -> int:
    """Estimate GPU usage (bytes) by running probe_fn in a fresh process.

    Returns peak allocated bytes (int). Raises RuntimeError on failure.
    """
    kwargs = kwargs or {}
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_probe_worker, args=(probe_fn, args, kwargs, q, gpu))
    p.start()
    try:
        ok, val = q.get(timeout=timeout)
    except Exception:
        p.terminate()
        p.join()
        raise RuntimeError("Probe process timed out or failed to report")
    p.join()
    if not ok:
        raise RuntimeError(f"Probe failed: {val}")
    return int(val)


def run_training_with_reservation(
    train_fn: Callable,
    train_args: Tuple = (),
    train_kwargs: dict = None,
    *,
    probe_fn: Optional[Callable] = None,
    probe_args: Tuple = (),
    probe_kwargs: dict = None,
    gpu: Optional[int] = None,
    countdown: Optional[float] = None,
    reserve_kwargs: dict = None,
    probe_timeout: float = 60.0,
) -> Any:
    """
    Run a training function with cooperative GPU reservation.

    Modes:
    - probe_fn provided: run the probe in a separate process to estimate peak GPU
      memory usage, then acquire reservation and run the real training in the
      current process inside the reservation context.
    - countdown provided (and no probe_fn): waits `countdown` seconds and then
      acquires reservation (no automatic estimation).

    Parameters:
    - train_fn: your main training callable. Will be executed inside the reservation context.
    - probe_fn: optional callable used to estimate GPU peak; should mimic allocations
      that the real training will perform (can be a short warmup).
    - gpu: GPU index to target; if None and CUDA available, current device is used.
    - reserve_kwargs: forwarded to reserve_gpu_for_training (e.g., safety_margin_bytes, blocking)

    Returns whatever train_fn returns.

    Notes: This is cooperative; for best results, ensure that the actual training
    run inside this function performs the heavy GPU allocations only after the
    reservation context is entered.
    """
    train_kwargs = train_kwargs or {}
    probe_kwargs = probe_kwargs or {}
    reserve_kwargs = reserve_kwargs or {}

    # If CUDA not available, just run training directly
    if not torch.cuda.is_available():
        return train_fn(*train_args, **train_kwargs)

    # determine GPU index
    if gpu is None:
        gpu_index = torch.cuda.current_device()
    else:
        gpu_index = gpu

    estimated = None
    if probe_fn is not None:
        estimated = estimate_gpu_usage_by_probe(
            probe_fn, probe_args, probe_kwargs, gpu_index, timeout=probe_timeout
        )
    elif countdown is not None:
        # optional short wait before reserving
        time.sleep(countdown)

    # If we have an estimate, pass training_reserved_bytes to reserve
    if estimated is not None:
        reserve_kwargs = dict(reserve_kwargs)
        reserve_kwargs.setdefault("training_reserved_bytes", int(estimated))

    # Acquire reservation and run training
    with reserve_gpu_for_training(gpu=gpu_index, **reserve_kwargs):
        return train_fn(*train_args, **train_kwargs)


def make_probe_fn(
    model_ctor: Optional[Callable] = None,
    input_shape: Optional[Tuple[int, ...]] = None,
    sample_batch: Optional[object] = None,
    device: Optional[str] = None,
    forward_and_backward: bool = False,
    warmup_steps: int = 1,
):
    """Return a probe function suitable for estimate_gpu_usage_by_probe.

    Usage patterns:
    - Provide a model constructor (callable, e.g. lambda: MyModel(...)) and
      an input_shape (e.g. (batch, C, H, W)). The returned probe will instantiate
      the model, move it to CUDA, create a random input tensor of the given shape
      and run forward (and optional backward) for `warmup_steps` times. This is
      lightweight and safe to run in the separate probe process.

    - Alternatively provide `sample_batch` (already-constructed tensor or tuple of tensors)
      which will be moved to device and used directly.

    Parameters:
    - model_ctor: callable that returns a freshly-constructed torch.nn.Module.
    - input_shape: shape tuple for a synthetic input tensor when sample_batch is None.
    - sample_batch: an example batch (tensor or tuple/list of tensors).
    - device: device string like 'cuda' or 'cuda:0'. If None, probe will use 'cuda' when available.
    - forward_and_backward: if True, do one backward pass (requires creating a small loss).
    - warmup_steps: number of forward (or forward+backward) iterations to run.

    The returned function takes no arguments and is safe to run in a spawn-ed process.
    """

    def probe():
        import torch

        if not torch.cuda.is_available():
            return

        dev = device or "cuda"

        # build model
        model = None
        if model_ctor is not None:
            try:
                model = model_ctor()
            except Exception:
                # try calling without args if a class was passed
                model = model_ctor
        if model is None:
            # nothing to move; we'll still run synthetic allocations
            pass
        else:
            try:
                model.to(dev)
                model.eval()
            except Exception:
                # best-effort; continue
                pass

        # prepare input
        if sample_batch is not None:
            batch = sample_batch
        elif input_shape is not None:
            # create a random float tensor and move to device
            batch = torch.randn(*input_shape, device=dev)
        else:
            # fallback small allocation
            batch = torch.empty(1024 * 1024 * 10, dtype=torch.uint8, device=dev)
            batch.fill_(1)

        # normalize batch to be a tensor or tuple/list of tensors
        if isinstance(batch, (list, tuple)):
            batch_tensors = batch
        else:
            batch_tensors = (batch,)

        # run warmup steps
        for _ in range(max(1, warmup_steps)):
            # forward
            out = None
            if model is not None:
                with torch.set_grad_enabled(forward_and_backward):
                    try:
                        out = model(*batch_tensors)
                    except Exception:
                        # some models expect a single tensor argument
                        try:
                            out = model(batch_tensors[0])
                        except Exception:
                            out = None
            else:
                # if no model, just touch tensors
                for t in batch_tensors:
                    if hasattr(t, "fill_"):
                        try:
                            t.fill_(0)
                        except Exception:
                            pass

            if forward_and_backward and model is not None:
                # create a small pseudo-loss and backward
                try:
                    # ensure out is a tensor
                    if isinstance(out, torch.Tensor):
                        loss = out.flatten().abs().sum()
                        loss.backward()
                    else:
                        # fallback: touch params
                        for p in model.parameters():
                            if p.requires_grad:
                                p.grad = torch.zeros_like(p)
                except Exception:
                    pass

        # synchronize to ensure allocations are realized
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    return probe
