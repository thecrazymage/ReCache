"""Construction of the diffusion model and its caching helper."""

import torch
from munch import Munch

from src.caching_helpers import get_cacher_helper, BaseCachingHelper
from src.diffusion_wrappers import get_diffusion_wrapper, BaseDiffusionWrapper


def get_models(
    wrapper_config: Munch, cacher_config: Munch, device: torch.device | str = torch.device('cuda')
) -> tuple[BaseDiffusionWrapper, BaseCachingHelper]:
    """Build a diffusion wrapper and the caching helper that wraps it.

    The helper class is resolved per model, e.g. `TaylorSeerHelper` for
    `FluxWrapper` becomes `TaylorSeerHelperFlux`.

    Args:
        wrapper_config: Config with `wrapper_class` and `parameters`.
        cacher_config: Config with `cacher_class` and optional
            `parameters`.
        device: Device to load the diffusion model on.

    Returns:
        Tuple `(model, caching_helper)`.
    """
    wrapper_class = wrapper_config.wrapper_class
    cacher_class = cacher_config.cacher_class

    model = get_diffusion_wrapper(
        wrapper_class, wrapper_config.parameters, device
    )
    helper = get_cacher_helper(
        cacher_class, cacher_config.get('parameters', {}), model
    )

    return model, helper
