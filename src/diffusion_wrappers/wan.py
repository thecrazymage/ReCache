"""Wan2.1-T2V-1.3B text-to-video wrapper."""

import torch
from munch import Munch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import (
    UniPCMultistepScheduler,
)

from src.diffusion_wrappers.base import BaseDiffusionWrapper


class WanWrapper(BaseDiffusionWrapper):
    """Wrapper around the Wan2.1-T2V-1.3B pipeline.

    Uses the UniPC multistep scheduler. The VAE runs in float32 and the
    rest of the pipeline in bfloat16, regardless of `dtype`.

    Config:
        nfe (int): Number of sampling steps.
        flow_shift (float, optional): Scheduler flow shift; 3.0 by
            default.
        width (int, optional): Video width; 832 by default.
        height (int, optional): Video height; 480 by default.
        num_frames (int, optional): Number of frames; 81 by default.
        guidance_scale (float, optional): Classifier-free guidance scale;
            5.0 by default.
        fps (int, optional): Frame rate of saved videos; 16 by default.
    """

    def __init__(
        self,
        config: Munch,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        super().__init__(config, dtype, device)

        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir='pretrained/models',
        )

        self.pipe = WanPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=torch.bfloat16,
            cache_dir='pretrained/models',
        ).to(self.device)

        self.pipe.set_progress_bar_config(disable=True)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            flow_shift=float(getattr(self.config, "flow_shift", 3.0)),
        )

        self.config.width = int(getattr(self.config, "width", 832))
        self.config.height = int(getattr(self.config, "height", 480))
        self.config.num_frames = int(getattr(self.config, "num_frames", 81))
        self.config.guidance_scale = float(
            getattr(self.config, 'guidance_scale', 5.0)
        )
        self.config.fps = int(getattr(self.config, "fps", 16))

        num_latent_frames = (
            self.config.num_frames - 1
        ) // self.pipe.vae_scale_factor_temporal + 1
        self.latents_shape = (
            16,
            num_latent_frames,
            self.config.height // self.pipe.vae_scale_factor_spatial,
            self.config.width // self.pipe.vae_scale_factor_spatial,
        )

        self.is_video = True

    def forward(
        self,
        prompts: list[str],
        latents: torch.Tensor | None = None,
        output_type: str = 'pt',
        *,
        output_video: bool = False,
    ) -> torch.Tensor:
        forward_args = dict(
            latents=latents,
            num_inference_steps=self.config.nfe,
            output_type=output_type,
            return_dict=False,
            height=self.config.height,
            width=self.config.width,
            num_frames=self.config.num_frames,
            guidance_scale=self.config.guidance_scale,
        )
        video = self.pipe(prompt=prompts, **forward_args)[0]

        if not output_video:
            video = video[:, 0]

        return video.to(torch.float32)
