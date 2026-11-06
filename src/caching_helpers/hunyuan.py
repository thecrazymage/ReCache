"""Caching helpers for `HunyuanWrapper`."""

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


class BaseCachingHelperHunyuan(BaseCachingHelper):
    """Base class for caching helpers of `HunyuanWrapper`.

    The pipeline calls the transformer once per sampling step, so the
    steps are counted by the transformer calls. By default, the
    attention layers of all blocks are wrapped, and their calls are
    routed to `_process_layer_output`.

    Attributes:
        cache: Cached data by layer key, see `_init_cache_structure`.
        last_timestep: Index of the last computed step.
    """

    def _init_cache_structure(self) -> Any:
        """Return an empty cache."""
        return defaultdict(list)

    def set_default(self) -> None:
        """Reset the cache and the step counter."""
        super().set_default()
        self.last_timestep = 0
        self.cache = self._init_cache_structure()

    def wrap_transformer(self) -> None:
        """Wrap the transformer to count the steps and set `recalculate`."""
        self.function_dict["transformer_forward"] = (
            self.model.pipe.transformer.forward
        )

        def wrapped_forward(*args, **kwargs):
            self.recalculate = self.timestep not in self.ts_to_skip
            out = self.function_dict["transformer_forward"](*args, **kwargs)

            if self.recalculate:
                self.last_timestep = self.timestep

            self.timestep += 1
            return out

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
        """Wrap the transformer and the attention layers of all blocks."""
        self.wrap_transformer()

        transformer = self.model.pipe.transformer

        for index, block in enumerate(transformer.transformer_blocks):
            self.wrap_layer(block.attn, ("dual_attn", index))

        for index, block in enumerate(transformer.single_transformer_blocks):
            self.wrap_layer(block.attn, ("single_attn", index))


class CacheAndForecastHelperHunyuan(
    BaseCachingHelperHunyuan, CacheAndForecastHelper
):
    """Base class for forecasting helpers of `HunyuanWrapper`.

    Each output of a cached layer is forecast separately.
    """

    def _init_cache_structure(self) -> Any:
        """Return an empty cache."""
        return defaultdict(dict)

    def _process_layer_output(
        self, index: Hashable, *args: Any, **kwargs: Any
    ) -> Any:
        """Compute the outputs of the layer or forecast them."""
        layer_cache = self.cache[index]

        if self.recalculate or len(layer_cache) == 0:
            outputs = self.function_dict[index](*args, **kwargs)
            self.cache[index] = {
                output_index: self.calculate_derivatives(
                    value,
                    layer_cache.get(output_index, []),
                    self.last_timestep,
                )
                for output_index, value in enumerate(outputs)
            }
            return outputs

        outputs = [
            self.predict_value(layer_cache[output_index], self.last_timestep)
            for output_index in range(len(layer_cache))
        ]
        return outputs


class FORACachingHelperHunyuan(BaseCachingHelperHunyuan):
    """FORA-style helper that reuses the attention outputs."""

    def _process_layer_output(
        self, index: Hashable, *args: Any, **kwargs: Any
    ) -> Any:
        """Compute the output of the layer or reuse the cached one."""
        layer_cache = self.cache[index]

        if self.recalculate or len(layer_cache) == 0:
            self.cache[index] = self.function_dict[index](*args, **kwargs)

        return self.cache[index]


class TaylorSeerHelperHunyuan(
    CacheAndForecastHelperHunyuan, TaylorSeerHelper
):
    """TaylorSeer helper for `HunyuanWrapper`."""


class HiCacheHelperHunyuan(CacheAndForecastHelperHunyuan, HiCacheHelper):
    """HiCache helper for `HunyuanWrapper`."""


class TaylorDPHelperHunyuan(BaseCachingHelperHunyuan, TaylorSeerHelper):
    """DPCache-style helper that forecasts the outputs of block stacks.

    The outputs are forecast with TaylorSeer. On skipped steps, all
    blocks pass their inputs through, except the last dual-stream and
    the last single-stream blocks, whose outputs are forecast. The two
    outputs of a block are concatenated along the sequence dimension and
    forecast as one tensor.
    """

    def _init_cache_structure(self) -> dict[str, list[torch.Tensor]]:
        """Return an empty cache for the two block stacks."""
        return {"dual": [], "single": []}

    def wrap_block(
        self, block: nn.Module, index: int, last_index: int, block_type: str
    ) -> None:
        """Wrap a transformer block.

        Args:
            block: Block to wrap.
            index: Index of the block in its stack.
            last_index: Index of the last block in the stack.
            block_type: Stack of the block: `'dual'` or `'single'`.
        """
        self.function_dict[(f"{block_type}_block", index)] = block.forward

        def wrapped_forward(*args, **kwargs):
            hidden_states, encoder_hidden_states = args[0], args[1]

            if self.recalculate:
                hidden_output, encoder_output = self.function_dict[
                    (f"{block_type}_block", index)
                ](*args, **kwargs)

                if index == last_index:
                    if block_type == "dual":
                        combined_output = torch.cat(
                            [encoder_output, hidden_output], dim=1
                        )
                    else:
                        combined_output = torch.cat(
                            [hidden_output, encoder_output], dim=1
                        )

                    self.cache[block_type] = self.calculate_derivatives(
                        combined_output,
                        self.cache[block_type],
                        self.last_timestep,
                    )

                return hidden_output, encoder_output

            if index == last_index:
                combined_output = self.predict_value(
                    self.cache[block_type], self.last_timestep
                )
                seq_len = encoder_hidden_states.shape[1]

                if block_type == "dual":
                    pred_hidden, pred_encoder = (
                        combined_output[:, seq_len:, :],
                        combined_output[:, :seq_len, :],
                    )
                else:
                    pred_hidden, pred_encoder = (
                        combined_output[:, :-seq_len, :],
                        combined_output[:, -seq_len:, :],
                    )

                return pred_hidden, pred_encoder

            return hidden_states, encoder_hidden_states

        block.forward = wrapped_forward

    def wrap_model(self) -> None:
        """Wrap the transformer and all blocks."""
        self.wrap_transformer()

        dual_blocks = self.model.pipe.transformer.transformer_blocks
        for index, block in enumerate(dual_blocks):
            self.wrap_block(block, index, len(dual_blocks) - 1, "dual")

        single_blocks = self.model.pipe.transformer.single_transformer_blocks
        for index, block in enumerate(single_blocks):
            self.wrap_block(block, index, len(single_blocks) - 1, "single")
