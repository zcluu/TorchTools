"""
Example that demonstrates how to probe and then run training with GPU reservation.
This example constructs a small probe that allocates some tensors to emulate peak usage.
"""

import time

import torch

from torchtools import (
    run_training_with_reservation,
    estimate_gpu_usage_by_probe,
    make_probe_fn,
)


# Example probe built from input shape (you can instead provide a model constructor)
probe_fn = make_probe_fn(input_shape=(16, 3, 224, 224), warmup_steps=1)


def train_fn(steps=5):
    print("Training start")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        x = torch.randn(1024, 1024, device=dev)  # some work tensor
        for i in range(steps):
            x = x * 0.99
            time.sleep(0.2)
            print("step", i)
    else:
        for i in range(steps):
            time.sleep(0.2)
            print("step", i)
    print("Training end")


if __name__ == "__main__":
    # Run probe and then training with reservation
    try:
        peak = estimate_gpu_usage_by_probe(probe_fn, timeout=30)
        print("Estimated peak bytes:", peak)
    except Exception as e:
        print("Probe failed:", e)
        peak = None

    run_training_with_reservation(
        train_fn,
        train_args=(5,),
        probe_fn=probe_fn,
        reserve_kwargs={"safety_margin_bytes": 20 * 1024**2},
    )
