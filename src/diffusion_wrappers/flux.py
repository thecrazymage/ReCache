"""FLUX.1-dev text-to-image wrapper."""

from typing import Any

import torch
from munch import Munch
from diffusers import FluxPipeline

from src.diffusion_wrappers.base import BaseDiffusionWrapper


class FluxWrapper(BaseDiffusionWrapper):
    """Wrapper around the FLUX.1-dev pipeline.

    Generates 1024 x 1024 images; `latents_shape` is the shape of the
    initial latents at this resolution.

    Config:
        nfe (int): Number of sampling steps.
        guidance_scale (float, optional): Embedded guidance scale;
            3.5 by default.
    """

    def __init__(
        self,
        config: Munch,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        super().__init__(config, dtype, device)

        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=self.dtype,
            cache_dir='pretrained/models',
        ).to(self.device)

        self.pipe.set_progress_bar_config(disable=True)

        self.config.guidance_scale = float(
            getattr(self.config, 'guidance_scale', 3.5)
        )
        self.latents_shape = (4096, 64)
        self.is_video = False

    def forward(
        self,
        prompts: list[str],
        latents: torch.Tensor | None = None,
        output_type: str = 'pt',
        **_: Any,
    ) -> torch.Tensor:
        return self.pipe(
            prompt=prompts,
            latents=latents,
            num_inference_steps=self.config.nfe,
            guidance_scale=self.config.guidance_scale,
            output_type=output_type,
            return_dict=False,
        )[0].to(torch.float32)
