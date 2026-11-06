"""Image and video metrics and the registry that maps names to them."""

from collections.abc import Callable
from functools import partial
from types import SimpleNamespace

from src.metrics.base import BaseMetric
import src.metrics.quality as quality
import src.metrics.similarity as similarity
import src.metrics.combined as combined

metrics_registry: dict[str, Callable[..., BaseMetric]] = {
    'LPIPS': similarity.LPIPSMetric,
    'PLPIPS': similarity.PLPIPSMetric,
    'L1': similarity.L1Metric,
    'SSIM': similarity.SSIMMetric,
    'PSNR': similarity.PSNRMetric,
    'HPS': quality.HPSMetric,
    'CLIP': quality.CLIPMetric,
    'IR': quality.IRMetric,
    'AQ': quality.AQMetric,
    'IQ': quality.IQMetric,
    'PLPIPS_HPS': combined.PLPIPS_HPS,
    'PLPIPS_7HPS': partial(
        combined.PLPIPS_HPS,
        config=SimpleNamespace(alpha_plpips=1.0, alpha_hps=-7.0),
    ),
}
