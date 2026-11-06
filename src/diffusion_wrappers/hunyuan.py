"""HunyuanVideo text-to-video wrapper."""

import torch
from munch import Munch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel

from src.diffusion_wrappers.base import BaseDiffusionWrapper


class HunyuanWrapper(BaseDiffusionWrapper):
    """Wrapper around the HunyuanVideo pipeline.

    The transformer runs in bfloat16 and the rest of the pipeline in
    `dtype`. Only the embedded guidance is used; true classifier-free
    guidance is disabled.

    Config:
        nfe (int): Number of sampling steps.
        width (int, optional): Video width; 640 by default.
        height (int, optional): Video height; 480 by default.
        num_frames (int, optional): Number of frames; 65 by default.
        guidance_scale (float, optional): Embedded guidance scale;
            6.0 by default.
        fps (int, optional): Frame rate of saved videos; 15 by default.
    """

    def __init__(
        self,
        config: Munch,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        super().__init__(config, dtype, device)

        model_id = "hunyuanvideo-community/HunyuanVideo"

        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            cache_dir='pretrained/models',
        )

        self.pipe = HunyuanVideoPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=self.dtype,
            cache_dir='pretrained/models',
        ).to(self.device)

        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.vae.enable_tiling()

        self.config.width = int(getattr(self.config, "width", 640))
        self.config.height = int(getattr(self.config, "height", 480))
        self.config.num_frames = int(getattr(self.config, "num_frames", 65))
        self.config.guidance_scale = float(
            getattr(self.config, 'guidance_scale', 6.0)
        )
        self.config.fps = int(getattr(self.config, "fps", 15))

        self.latents_shape = (
            int(self.pipe.transformer.config.in_channels),
            (self.config.num_frames - 1)
            // int(self.pipe.vae.temporal_compression_ratio)
            + 1,
            self.config.height // int(self.pipe.vae.spatial_compression_ratio),
            self.config.width // int(self.pipe.vae.spatial_compression_ratio),
        )
        self.is_video = True

    def forward(
        self,
        prompts: list[str],
        latents: torch.Tensor | None = None,
        output_type: str = "pt",
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
            true_cfg_scale=1.0,
        )

        video = self.pipe(prompt=prompts, **forward_args)[0]

        if not output_video:
            video = video[:, 0]

        return video.to(torch.float32)
