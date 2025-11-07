import time
from typing import Any, Callable, Optional, Tuple
import multiprocessing as mp

import torch
import importlib

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


class ProbeFunction:
    """Top-level picklable probe callable used by the spawn'ed worker.

    model_ctor may be either a callable (must be picklable / importable) or a
    string of the form 'module:callable' which will be resolved in the child
    process. Other parameters are simple data and therefore picklable.
    """

    def __init__(
        self,
        model_ctor: Optional[Callable] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        sample_batch: Optional[object] = None,
        device: Optional[str] = None,
        forward_and_backward: bool = False,
        warmup_steps: int = 1,
    ):
        self.model_ctor = model_ctor
        self.input_shape = input_shape
        self.sample_batch = sample_batch
        self.device = device
        self.forward_and_backward = forward_and_backward
        self.warmup_steps = int(warmup_steps)

    def _resolve_model_ctor(self):
        # If model_ctor is a string 'module:attr', import it in the child
        if isinstance(self.model_ctor, str):
            module, sep, attr = self.model_ctor.partition(":")
            if not sep:
                raise ValueError("model_ctor string must be 'module:callable'")
            mod = importlib.import_module(module)
            return getattr(mod, attr)
        return self.model_ctor

    def __call__(self):
        # The actual probe code runs here inside the spawned process
        import torch as _torch

        if not _torch.cuda.is_available():
            return

        dev = self.device or "cuda"

        model = None
        model_ctor = self._resolve_model_ctor()
        if model_ctor is not None:
            try:
                model = model_ctor()
            except Exception:
                # model_ctor may be a class rather than a constructor
                model = model_ctor

        if model is not None:
            try:
                model.to(dev)
                model.eval()
            except Exception:
                pass

        # prepare input
        if self.sample_batch is not None:
            batch = self.sample_batch
            # try to move tensors to device if they are tensors
            if isinstance(batch, (_torch.Tensor,)):
                try:
                    batch = batch.to(dev)
                except Exception:
                    pass
        elif self.input_shape is not None:
            batch = _torch.randn(*self.input_shape, device=dev)
        else:
            batch = _torch.empty(1024 * 1024 * 10, dtype=_torch.uint8, device=dev)
            batch.fill_(1)

        if isinstance(batch, (list, tuple)):
            batch_tensors = batch
        else:
            batch_tensors = (batch,)

        for _ in range(max(1, self.warmup_steps)):
            out = None
            if model is not None:
                with _torch.set_grad_enabled(self.forward_and_backward):
                    try:
                        out = model(*batch_tensors)
                    except Exception:
                        try:
                            out = model(batch_tensors[0])
                        except Exception:
                            out = None
            else:
                for t in batch_tensors:
                    if hasattr(t, "fill_"):
                        try:
                            t.fill_(0)
                        except Exception:
                            pass

            if self.forward_and_backward and model is not None:
                try:
                    if isinstance(out, _torch.Tensor):
                        loss = out.flatten().abs().sum()
                        loss.backward()
                    else:
                        for p in model.parameters():
                            if p.requires_grad:
                                p.grad = _torch.zeros_like(p)
                except Exception:
                    pass

        try:
            _torch.cuda.synchronize()
        except Exception:
            pass


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

    # Return a top-level picklable ProbeFunction instance
    return ProbeFunction(
        model_ctor=model_ctor,
        input_shape=input_shape,
        sample_batch=sample_batch,
        device=device,
        forward_and_backward=forward_and_backward,
        warmup_steps=warmup_steps,
    )
