"""Caching helpers for `WanWrapper`."""

from collections import defaultdict
from collections.abc import Hashable
from typing import Any

import torch
import torch.nn as nn

from src.caching_helpers.base import (
    BaseCachingHelper,
    CacheAndForecastHelper,
    TaylorSeerHelper,
    HiCacheHelper,
)


class BaseCachingHelperWan(BaseCachingHelper):
    """Base class for caching helpers of `WanWrapper`.

    With classifier-free guidance, the pipeline calls the transformer
    twice per sampling step, and each call of a step has its own cache
    slot. The step index is looked up from the scheduler timestep. By
    default, the attention and feed-forward layers of all blocks are
    wrapped, and their calls are routed to `_process_layer_output`.

    Attributes:
        cache: Cached data by layer key and slot, see
            `_init_cache_structure`.
        last_timestep: Index of the last computed step, per slot.
    """

    def _init_cache_structure(self) -> Any:
        """Return an empty cache with two slots per layer."""
        return defaultdict(lambda: [[], []])

    def set_default(self) -> None:
        """Reset the cache and the step tracking."""
        super().set_default()
        self.last_timestep = [0, 0]
        self._prev_timestep = None
        self._call_in_step = 0
        self._cfg_slot = 0
        self.cache = self._init_cache_structure()

    def wrap_transformer(self) -> None:
        """Wrap the transformer to track the step and the CFG slot."""
        self.function_dict["transformer_forward"] = (
            self.model.pipe.transformer.forward
        )

        def wrapped_forward(*args, **kwargs):
            self.timestep = list(self.model.pipe.scheduler.timesteps).index(
                kwargs["timestep"].flatten()[0].item()
            )

            if self._prev_timestep != self.timestep:
                self._prev_timestep = self.timestep
                self._call_in_step = 0
            else:
                self._call_in_step += 1

            self._cfg_slot = 0 if self._call_in_step == 0 else 1
            self.recalculate = self.timestep not in self.ts_to_skip

            output = self.function_dict["transformer_forward"](*args, **kwargs)

            if self.recalculate:
                self.last_timestep[self._cfg_slot] = self.timestep

            return output

        self.model.pipe.transformer.forward = wrapped_forward

    def wrap_layer(self, layer: nn.Module, index: Hashable) -> None:
        """Route the calls of the layer to `_process_layer_output`."""
        self.function_dict[index] = layer.forward

        def wrapped_forward(*args, **kwargs):
            return self._process_layer_output(index, *args, **kwargs)

        layer.forward = wrapped_forward

    def _process_layer_output(
        self, index: Hashable, *args: Any, **kwargs: Any
    ) -> Any:
        """Compute or reuse the output of a wrapped layer.

        Args:
            index: Key of the layer; the other arguments are passed to
                the layer.

        Returns:
            Output of the layer.
        """
        raise NotImplementedError()

    def wrap_model(self) -> None:
        """Wrap the transformer and the cached layers of all blocks."""
        self.wrap_transformer()

        for index, block in enumerate(self.model.pipe.transformer.blocks):
            self.wrap_layer(block.attn1, ("attn1", index))
            self.wrap_layer(block.attn2, ("attn2", index))
            self.wrap_layer(block.ffn, ("ffn", index))


class FORACachingHelperWan(BaseCachingHelperWan):
    """FORA-style helper that reuses the attention and feed-forward outputs."""

    def _process_layer_output(
        self, index: Hashable, *args: Any, **kwargs: Any
    ) -> Any:
        """Compute the output of the layer or reuse the cached one."""
        slot = self._cfg_slot

        if self.recalculate:
            self.cache[index][slot] = self.function_dict[index](
                *args, **kwargs
            )

        return self.cache[index][slot]


class CacheAndForecastHelperWan(BaseCachingHelperWan, CacheAndForecastHelper):
    """Base class for forecasting helpers of `WanWrapper`."""

    def _process_layer_output(
        self, index: Hashable, *args: Any, **kwargs: Any
    ) -> Any:
        """Compute the output of the layer or forecast it."""
        slot = self._cfg_slot

        if self.recalculate:
            output = self.function_dict[index](*args, **kwargs)
            self.cache[index][slot] = self.calculate_derivatives(
                output,
                self.cache[index][slot],
                self.last_timestep[slot],
            )
            return output
        else:
            return self.predict_value(
                self.cache[index][slot], self.last_timestep[slot]
            )


class TaylorSeerHelperWan(CacheAndForecastHelperWan, TaylorSeerHelper):
    """TaylorSeer helper for `WanWrapper`."""


class HiCacheHelperWan(CacheAndForecastHelperWan, HiCacheHelper):
    """HiCache helper for `WanWrapper`."""


class TaylorDPHelperWan(BaseCachingHelperWan, TaylorSeerHelper):
    """DPCache-style helper that forecasts the output of the last block.

    The output is forecast with TaylorSeer. On skipped steps, all the
    other blocks pass their inputs through.
    """

    def _init_cache_structure(self) -> list[list[torch.Tensor]]:
        """Return an empty cache with two slots."""
        return [[], []]

    def wrap_block(
        self, block: nn.Module, index: int, last_index: int
    ) -> None:
        """Wrap a transformer block.

        Args:
            block: Block to wrap.
            index: Index of the block.
            last_index: Index of the last block.
        """
        self.function_dict[("block", index)] = block.forward

        def wrapped_forward(
            hidden_states, encoder_hidden_states, *args, **kwargs
        ):
            slot = self._cfg_slot

            if self.recalculate:
                output = self.function_dict[("block", index)](
                    hidden_states,
                    encoder_hidden_states,
                    *args,
                    **kwargs,
                )

                if index == last_index:
                    self.cache[slot] = self.calculate_derivatives(
                        output,
                        self.cache[slot],
                        self.last_timestep[slot],
                    )

                return output

            if index == last_index:
                return self.predict_value(
                    self.cache[slot], self.last_timestep[slot]
                )

            return hidden_states

        block.forward = wrapped_forward

    def wrap_model(self) -> None:
        """Wrap the transformer and all blocks."""
        self.wrap_transformer()

        blocks = self.model.pipe.transformer.blocks
        last_index = len(blocks) - 1

        for index, block in enumerate(blocks):
            self.wrap_block(block, index, last_index)
