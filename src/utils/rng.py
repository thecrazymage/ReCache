"""Seeding and random sampling helpers."""

import random

import torch
import numpy as np
from munch import Munch


def seed_everything(seed: int = 42) -> None:
    """Seed all RNGs and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weighted_randint(count: int, config: Munch) -> torch.Tensor:
    """Sample integers from a range with an emphasized subrange.

    Each value in `[min_val, max_val]` gets weight `weight`, except
    values in the emphasized subrange `[min_emp_val, max_emp_val]`,
    which get `emp_weight`. Weights are unnormalized.

    Args:
        count: Number of samples.
        config: Sampling config, see `Config`.

    Config:
        min_val (int): Lower bound, inclusive.
        max_val (int): Upper bound, inclusive.
        weight (float): Weight of values outside the emphasized subrange.
        min_emp_val (int): Lower bound of the emphasized subrange,
            inclusive.
        max_emp_val (int): Upper bound of the emphasized subrange,
            inclusive.
        emp_weight (float): Weight of values in the emphasized subrange.

    Returns:
        Sampled integers of shape (count,).
    """

    num_classes = config.max_val - config.min_val + 1
    weights = torch.full((num_classes,), config.weight)

    idx_start = max(0, config.min_emp_val - config.min_val)
    idx_end = min(num_classes - 1, config.max_emp_val - config.min_val)
    weights[idx_start:idx_end + 1] = config.emp_weight

    indices = torch.multinomial(weights, count, replacement=True)
    return indices + config.min_val
