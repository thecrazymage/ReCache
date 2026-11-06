"""Diffusion pipeline wrappers and the factory that builds them."""

import torch
from munch import Munch

from src.diffusion_wrappers.base import BaseDiffusionWrapper
from src.diffusion_wrappers.flux import FluxWrapper
from src.diffusion_wrappers.hunyuan import HunyuanWrapper
from src.diffusion_wrappers.wan import WanWrapper


def get_diffusion_wrapper(
    wrapper_class: str, config: Munch, device: torch.device | str = torch.device('cuda')
) -> BaseDiffusionWrapper:
    """Build the diffusion wrapper named `wrapper_class`.

    Raises:
        NotImplementedError: If the wrapper class is unknown.
    """
    if wrapper_class == 'FluxWrapper':
        return FluxWrapper(config, device=device)

    if wrapper_class == 'HunyuanWrapper':
        return HunyuanWrapper(config, device=device)

    if wrapper_class == 'WanWrapper':
        return WanWrapper(config, device=device)

    raise NotImplementedError(
        f'unknown wrapper helper class {wrapper_class} (('
    )
