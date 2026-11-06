"""Metrics combined from several weighted metrics."""

from types import SimpleNamespace

import torch
from munch import Munch

from src.metrics.base import BaseMetric
from src.metrics.quality import HPSMetric
from src.metrics.similarity import PLPIPSMetric


class CombinedMetric(BaseMetric):
    """Weighted sum of several metrics.

    Args:
        config: Metric config.
        metrics: Metrics to combine.
        weights: Weight of each metric.

        See `BaseMetric` for the other arguments.
    """

    def __init__(
        self,
        config: Munch | SimpleNamespace | None,
        metrics: list[BaseMetric],
        weights: list[float],
        is_train: bool = False,
        is_video: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        super().__init__(config, is_train, is_video, dtype, device)
        self.metrics = metrics
        self.weights = weights

    @torch.inference_mode()
    def _calculate_batch(
        self,
        generated: torch.Tensor,
        references: torch.Tensor | None = None,
        prompts: str | list[str] | None = None,
    ) -> torch.Tensor:
        total_score = 0.0

        for metric, weight in zip(self.metrics, self.weights):
            score = metric._calculate_batch(
                generated=generated, references=references, prompts=prompts
            )
            total_score += weight * score

        return total_score

    @torch.inference_mode()
    def calculate_dataset(
        self,
        generated: torch.Tensor,
        references: torch.Tensor | None = None,
        prompts: str | list[str] | None = None,
        batch_size: int = 1,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        total_score = 0.0

        for metric, weight in zip(self.metrics, self.weights):
            score = metric.calculate_dataset(
                generated=generated,
                references=references,
                prompts=prompts,
                batch_size=batch_size,
                reduction=reduction,
            )
            total_score += weight * score

        return total_score


class PLPIPS_HPS(CombinedMetric):
    """`alpha_plpips * PLPIPS + alpha_hps * HPS`.

    With `is_train=True`, HPS is negated, so a positive `alpha_hps` gives
    `PLPIPS - alpha_hps * HPS`. For evaluation, a negative `alpha_hps`
    gives the same quantity, as in `PLPIPS_7HPS` from `metrics_registry`.

    Args:
        config: Metric config, see `Config`.
        patch_size: PLPIPS patch size, see `PLPIPSMetric`.
        stride: PLPIPS patch stride.
        lpips_net: LPIPS backbone.

        See `BaseMetric` for the other arguments.

    Config:
        alpha_plpips (float): PLPIPS weight.
        alpha_hps (float): HPS weight.
    """

    def __init__(
        self,
        config: Munch | SimpleNamespace,
        is_train: bool = False,
        is_video: bool = False,
        patch_size: int = 224,
        stride: int = 160,
        lpips_net: str = 'vgg',
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        plpips_metric = PLPIPSMetric(
            config,
            is_train,
            is_video,
            patch_size=patch_size,
            stride=stride,
            net=lpips_net,
            device=device,
            dtype=dtype,
        )
        hps_metric = HPSMetric(
            config, is_train, is_video, device=device, dtype=dtype
        )

        super().__init__(
            config,
            metrics=[plpips_metric, hps_metric],
            weights=[config.alpha_plpips, config.alpha_hps],
            is_train=is_train,
            is_video=is_video,
            device=device,
            dtype=dtype,
        )

    def __str__(self) -> str:
        alpha_plpips, alpha_hps = self.weights
        return f'{alpha_plpips} * PLPIPS + {alpha_hps} * HPS'
