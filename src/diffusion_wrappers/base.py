"""Base class for diffusion pipeline wrappers."""

import torch
import torch.nn as nn
from munch import Munch


class BaseDiffusionWrapper(nn.Module):
    """Base class for wrappers around diffusers text-to-image/video pipelines.

    A wrapper runs the full sampling loop with `config.nfe` steps for a
    batch of prompts from the given initial latents.

    Args:
        config: Wrapper config; every wrapper requires `nfe`, the number
            of sampling steps.
        dtype: Pipeline dtype.
        device: Device to run the pipeline on.

    Attributes:
        latents_shape: Shape of the initial latents of one sample.
        is_video: Whether the model generates videos.
    """

    latents_shape: tuple[int, ...]
    is_video: bool

    def __init__(
        self,
        config: Munch,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        prompts: list[str],
        latents: torch.Tensor | None = None,
        output_type: str = 'pt',
        *,
        output_video: bool = False,
    ) -> torch.Tensor:
        """Generate images or videos for a batch of prompts.

        Args:
            prompts: Batch of prompts.
            latents: Initial latents of shape (B, *latents_shape); sampled
                by the pipeline if None.
            output_type: Output type of the diffusers pipeline.
            output_video: For video models, whether to return full videos
                instead of their first frames. Ignored by image models.

        Returns:
            Float32 images of shape (B, C, H, W) in [0, 1], or videos of
            shape (B, T, C, H, W) if `output_video` is set.
        """
        raise NotImplementedError()
