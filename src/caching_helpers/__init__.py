"""Caching helpers that implement chosen caching mechanisms."""

from munch import Munch

import src.caching_helpers.flux as flux
import src.caching_helpers.hunyuan as hunyuan
import src.caching_helpers.wan as wan
from src.caching_helpers.base import BaseCachingHelper
from src.diffusion_wrappers import (
    BaseDiffusionWrapper,
    FluxWrapper,
    HunyuanWrapper,
    WanWrapper,
)


def get_cacher_helper(
    cacher_class: str, config: Munch, model: BaseDiffusionWrapper
) -> BaseCachingHelper:
    """Build a caching helper and wrap the model with it.

    Args:
        cacher_class: Model-agnostic helper name, e.g.
            `TaylorSeerHelper`; the model suffix (`Flux`, `Hunyuan` or
            `Wan`) is appended to it.
        config: Helper config, see the helper class.
        model: Diffusion wrapper to wrap in place.

    Returns:
        Helper that has wrapped `model`.

    Raises:
        NotImplementedError: If the model class is unknown.
    """
    if isinstance(model, FluxWrapper):
        helper_class = getattr(flux, cacher_class + 'Flux')
    elif isinstance(model, HunyuanWrapper):
        helper_class = getattr(hunyuan, cacher_class + 'Hunyuan')
    elif isinstance(model, WanWrapper):
        helper_class = getattr(wan, cacher_class + 'Wan')
    
    else:
        raise NotImplementedError(
            f'unknown model class {type(model).__name__}'
        )

    return helper_class(model, config)
