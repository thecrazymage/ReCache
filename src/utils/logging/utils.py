"""Helpers for logged images and plots and for computing test metrics."""

from collections.abc import Sequence
from itertools import chain
from typing import Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from matplotlib.figure import Figure
from torchvision.utils import make_grid

from src.metrics import metrics_registry
from src.utils.ddp import ddp_reduce_sum
from src.utils.memory import flush


def prepare_images(
    teacher_images: torch.Tensor,
    generated_images: torch.Tensor,
    log_images_size: int = 512,
) -> np.ndarray:
    """Build a two-column grid of reference and generated images.

    Row `i` holds `teacher_images[i]` on the left and `generated_images[i]`
    on the right. The grid is resized so that each cell is roughly
    `log_images_size` x `log_images_size`.

    Args:
        teacher_images: Reference images of shape (K, C, H, W) in [0, 1].
        generated_images: Generated images of the same shape.
        log_images_size: Side of one grid cell in pixels.

    Returns:
        Grid of shape (K * log_images_size, 2 * log_images_size, C)
        and dtype uint8.
    """
    log_images_k = len(teacher_images)
    assert (
        len(generated_images) == log_images_k
    ), "Generated and teacher images lists do not have the same size"

    images_shape = teacher_images[0].shape
    assert all(
        i.shape == images_shape
        for i in chain(teacher_images, generated_images)
    ), "All tensors must have the same shape"

    images = [
        torch.stack([t, g], dim=0)
        for t, g in zip(teacher_images, generated_images)
    ]
    images = torch.concat(images, dim=0)

    grid = make_grid(images, nrow=2, padding=4, pad_value=1.0)

    new_height = log_images_size * log_images_k
    new_width = log_images_size * 2
    resized_grid = F.resize(grid, (new_height, new_width), antialias=True)
    return (
        (resized_grid.permute(1, 2, 0) * 255)
        .round()
        .clamp(0, 255)
        .byte()
        .numpy()
    )


def reverse_timesteps(
    timesteps: Sequence[int] | torch.Tensor, nfes: int
) -> tuple[list[bool], list[int]]:
    """Complement a set of steps within `range(nfes)`.

    Args:
        timesteps: Step indices to exclude.
        nfes: Total number of steps.

    Returns:
        Tuple `(mask, indices)`, where `mask[t]` is True iff step `t` is
        not in `timesteps`, and `indices` lists such steps in order.
    """
    steps = [t not in timesteps for t in range(nfes)]
    ones_positions = [i for i, val in enumerate(steps) if val == 1]
    return steps, ones_positions


def draw_timesteps_plot(
    ts_to_skip: Sequence[int] | torch.Tensor, nfes: int, step: int
) -> Figure:
    """Plot which diffusion steps are computed and which are skipped.

    Computed steps are marked with red dots and dashed vertical lines.

    Args:
        ts_to_skip: Indices of skipped steps.
        nfes: Total number of steps.
        step: Value shown in the figure title.

    Returns:
        Matplotlib figure; the caller is responsible for closing it.
    """
    fig, ax = plt.subplots(1, 1, figsize=(nfes // 4, 3))

    steps, ones_positions = reverse_timesteps(ts_to_skip, nfes)
    ax.plot(range(nfes), steps)

    for pos in ones_positions:
        ax.axvline(x=pos, ymax=1, linestyle='--', color='green', alpha=0.7)
        ax.scatter(pos, 1, color='red', s=50, zorder=10)

    ax.set_xticks(range(nfes))

    fig.suptitle(f'Timesteps, step {step}')
    plt.tight_layout()
    return fig


def calc_metrics(
    metric_names: Sequence[str],
    is_video: bool,
    device: torch.device | str,
    **data: Any,
) -> dict[str, float]:
    """Compute dataset-mean metrics, averaged over all ranks.

    Each metric is created from `metrics_registry`, evaluated on the
    local shard and freed right after. Must be called on all ranks
    under DDP.

    Args:
        metric_names: Keys in `metrics_registry`.
        is_video: Whether `generated` holds videos.
        device: Device to run the metrics on.
        **data: Arguments of `BaseMetric.calculate_dataset`; must
            include `generated` and `prompts`.

    Returns:
        Mapping from `metrics/<name>` to the metric value.
    """
    results = {}

    for metric_name in metric_names:
        metricInstance = metrics_registry[metric_name](
            device=device, is_video=is_video
        )

        result = metricInstance.calculate_dataset(**data, reduction='sum')

        sz = torch.tensor(len(data['prompts']), device=result.device)
        result = ddp_reduce_sum(result) / ddp_reduce_sum(sz)

        results[f"metrics/{metric_name}"] = result.item()

        del metricInstance
        flush()

    return results


def to_video_frames(x: torch.Tensor) -> list[np.ndarray]:
    """Convert a video tensor to frames for `export_to_video`.

    Args:
        x: Video of shape (T, C, H, W), or (B, T, C, H, W), in which case
            only the first video is used. Values are in [0, 1] or
            [-1, 1]; the latter is detected by any negative value.

    Returns:
        List of T float frames of shape (H, W, C) in [0, 1].
    """
    x = x[0] if x.dim() == 5 else x  # -> (T, C, H, W)
    x = x.detach().float().cpu()
    if x.min().item() < 0:  # if [-1, 1] -> [0, 1]
        x = (x + 1) / 2
    x = x.clamp(0, 1).permute(0, 2, 3, 1).numpy()  # -> (T, H, W, 3)
    return list(x)
