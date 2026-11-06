"""GPU memory helpers."""

import gc

import torch


def flush() -> None:
    """Run garbage collection and free the CUDA memory cache."""
    gc.collect()
    torch.cuda.empty_cache()
