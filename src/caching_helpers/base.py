"""Base caching helpers."""

import math
from collections.abc import Callable, Hashable, Generator
from contextlib import contextmanager
from typing import Any

import torch
from munch import Munch

from src.diffusion_wrappers import BaseDiffusionWrapper


class BaseCachingHelper:
    """Base class for helpers that skip transformer computations.

    On creation, the helper wraps the transformer of `model` and some of
    its layers in place by replacing their `forward`. Outside `inference`
    every sampling step is computed; inside it, the steps from
    `ts_to_skip` reuse or forecast cached layer outputs instead of
    computing them. Subclasses implement `wrap_model` for a specific
    caching method and a model.

    Args:
        model: Diffusion wrapper to wrap.
        config: Helper config; the keys depend on the caching method.

    Attributes:
        function_dict: Original `forward` of each wrapped module, by key.
        cached_output: Cached outputs, by key.
        recalculate: Whether the current step is computed.
        ts_to_skip: Indices of the sampling steps to skip.
        timestep: Index of the current sampling step.
    """

    def __init__(self, model: BaseDiffusionWrapper, config: Munch) -> None:
        self.config = config
        self.model = model

        self.function_dict: dict[Hashable, Callable[..., Any]] = dict()

        self.set_default()
        self.wrap_model()

    @contextmanager
    def inference(
        self, ts_to_skip: list[int] | torch.Tensor
    ) -> Generator[None]:
        """Skip the given sampling steps inside the context.

        The cache and the step counter are reset on entering and exiting,
        so run exactly one generation per context.

        Args:
            ts_to_skip: 0-based indices of the sampling steps to skip.
        """
        self.set_default()

        try:
            if torch.is_tensor(ts_to_skip):
                ts_to_skip = ts_to_skip.detach().cpu().tolist()
            self.ts_to_skip = set(map(int, ts_to_skip))
            yield
        
        finally:
            self.set_default()

    def set_default(self) -> None:
        """Reset the cache and the step counter."""
        self.recalculate = True
        self.cached_output: dict[Hashable, Any] = dict()
        self.ts_to_skip = []
        self.timestep = 0

    def wrap_model(self) -> None:
        """Wrap the transformer and the cached layers of the model."""
        raise NotImplementedError()


class CacheAndForecastHelper(BaseCachingHelper):
    """Base class for helpers that forecast the skipped outputs.

    On computed steps, a cached layer stores its output with finite
    difference estimates of its derivatives over the step index; on
    skipped steps, `predict_value` forecasts the output from them.

    Config:
        order (int): Maximum order of the estimated derivatives.
    """

    def calculate_derivatives(
        self,
        value: torch.Tensor,
        cached_d: list[torch.Tensor],
        last_timestep: int,
    ) -> list[torch.Tensor]:
        """Update the derivative estimates with a newly computed output.

        Args:
            value: Layer output at the current step.
            cached_d: Output and its derivative estimates
                `[f, f', f'', ...]` at the last computed step; empty
                before the first one.
            last_timestep: Index of the last computed step.

        Returns:
            Output and its derivative estimates at the current step, up
            to order `config.order`.
        """
        iterations = min(len(cached_d), self.config.order)

        derivatives = [value]
        ts_difference = self.timestep - last_timestep

        for i in range(iterations):
            derivatives.append((derivatives[-1] - cached_d[i]) / ts_difference)

        return derivatives

    def predict_value(
        self, cached_d: list[torch.Tensor], last_timestep: int
    ) -> torch.Tensor:
        """Forecast the layer output at the current step.

        Args:
            cached_d: Output and its derivative estimates at the last
                computed step, see `calculate_derivatives`.
            last_timestep: Index of the last computed step.

        Returns:
            Forecast output.
        """
        raise NotImplementedError()


class TaylorSeerHelper(CacheAndForecastHelper):
    """TaylorSeer helper that forecasts the outputs with a Taylor series.

    See `CacheAndForecastHelper` for the config.
    """

    def predict_value(
        self, cached_d: list[torch.Tensor], last_timestep: int
    ) -> torch.Tensor:
        """Extrapolate the output with a Taylor series."""
        ts_difference = self.timestep - last_timestep

        prediction = cached_d[0].clone()
        for i in range(1, len(cached_d)):
            value = cached_d[i]
            prediction += (
                value * math.pow(ts_difference, i) / math.factorial(i)
            )

        return prediction


class HiCacheHelper(CacheAndForecastHelper):
    """HiCache helper that forecasts the outputs with Hermite polynomials.

    The code is adapted from
    https://github.com/fenglang918/HiCache/blob/main/models/hicache_fast_impl.py
    with minimal changes to be compatible with `CacheAndForecastHelper`.

    Config:
        order (int): Maximum order of the estimated derivatives.
        scale_factor (float): Scale `s` of the expansion: the k-th term
            is the k-th derivative estimate times `H_k(s * dt) s^k / k!`,
            where `dt` is the number of steps since the last computed one.
    """

    def _hicache_polynomial(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Hermite: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
        """
        if n == 0:
            return torch.ones_like(x)
        if n == 1:
            return 2 * x

        H_prev = torch.ones_like(x)
        H_curr = 2 * x

        for k in range(2, n + 1):
            H_next = 2 * x * H_curr - 2 * (k - 1) * H_prev
            H_prev, H_curr = H_curr, H_next

        return H_curr

    def predict_value(
        self, cached_d: list[torch.Tensor], last_timestep: int
    ) -> torch.Tensor:
        """
        基于 Hermite 多项式的特征预测
        """
        x = self.timestep - last_timestep
        feats_d = cached_d
        order = len(feats_d) - 1

        if order < 1:
            return feats_d[0]

        F_latest = feats_d[0].clone()
        x_tensor = torch.tensor(
            float(x), dtype=F_latest.dtype, device=F_latest.device
        )

        x_scaled = x_tensor * self.config.scale_factor

        pred = F_latest.clone()

        for k in range(1, order + 1):
            diff_k = feats_d[k]
            Hk = self._hicache_polynomial(x_scaled, k)
            alpha = float(Hk / math.factorial(k)) * (
                self.config.scale_factor**k
            )
            pred.add_(diff_k, alpha=alpha)

        return pred
