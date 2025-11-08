# TorchTools

TorchTools is a small utility package for PyTorch, providing helpers originally. It focuses on device selection and cooperative GPU reservation/scheduling for more reliable training on shared machines.

Key features
- Automatic device selection (CUDA / MPS / CPU) based on environment and PyTorch build.
- Cooperative GPU reservation using a filesystem lock plus an optional "fence" tensor that occupies spare GPU memory to prevent other cooperative processes from grabbing it.
- Probe functions to estimate peak GPU memory for a training job, allowing automatic tuning of reservation sizes.

Requirements
- Python >= 3.8
- torch

Installation

Install in editable mode for development:

```bash
pip install -e .
```

Or install the package directly:

```bash
pip install .
```

Quick start

Two common workflows are supported: (1) reserve a GPU with the context manager before building/starting your training, or (2) run a short probe to estimate peak memory and then run training with an automatic reservation.

Example — probe then reserve and run training

```python
from torchtools import run_training_with_reservation, make_probe_fn
import torch
import time

# Build a lightweight probe (you can also provide a model constructor or sample_batch)
probe_fn = make_probe_fn(input_shape=(16, 3, 224, 224), warmup_steps=1)

def train_fn(steps=5):
    print("Training start")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        x = torch.randn(1024, 1024, device=dev)
        for i in range(steps):
            x = x * 0.99
            time.sleep(0.2)
            print("step", i)
    else:
        for i in range(steps):
            time.sleep(0.2)
            print("step", i)
    print("Training end")

run_training_with_reservation(
    train_fn,
    train_args=(5,),
    probe_fn=probe_fn,
    reserve_kwargs={"safety_margin_bytes": 20 * 1024**2},
)
```

Example — manually reserve using the file-lock + fence context manager

```python
from torchtools import reserve_gpu_for_training
import torch

with reserve_gpu_for_training(gpu=0, training_reserved_bytes=6 * 1024**3):
    # Build models / run training here while the reservation is active
    pass
```

API overview

- get_device_from_env() — choose and return a torch.device based on environment and available backends (see `torchtools.device`).
- reserve_gpu_for_training(gpu=None, training_reserved_bytes=0, safety_margin_bytes=..., blocking=True, timeout=None, fence=True) — cooperative GPU reservation context manager (see `torchtools.gpu_lock`).
- estimate_gpu_usage_by_probe(probe_fn, ...) — run a probe in a fresh process and return an estimated peak memory usage in bytes.
- make_probe_fn(...) — helper to create a picklable probe callable (see `torchtools.gpu_scheduler`).
- run_training_with_reservation(train_fn, ..., probe_fn=..., reserve_kwargs=...) — full flow that optionally probes and then runs training inside a reservation.

Environment variables
- TORCHTOOLS_LOCK_DIR: directory used to store filesystem lock files (default: `/tmp`).

Notes and limitations

- This package uses a cooperative strategy: file locks and fence tensors coordinate between processes that use these helpers. It does not prevent unrelated processes from using the GPU.
- The fence tensor is best-effort. If allocation fails due to fragmentation or insufficient memory, the code falls back to the filesystem lock alone.
- Probe functions run in a spawned child process (multiprocessing spawn). Probe callables must be picklable/importable in the child process (prefer top-level callables or provide sample_batch).

Contributing & License

- Contributions are welcome via issues and pull requests. The project is distributed under the MIT License (see LICENSE).

More information

- See `examples/run_with_reservation_example.py` for a runnable example demonstrating probe + reservation + training.

---

Package info
- Name: torchtools
- Version: see package metadata (`torchtools.src.version` and package metadata in `pyproject.toml`).

If you'd like a more detailed API reference, type signatures, or additional examples (e.g., multi-GPU or integration with distributed training), tell me which parts to expand and I'll add them.
