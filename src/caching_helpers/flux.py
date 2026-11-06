"""Caching helpers for `FluxWrapper`."""

from collections import defaultdict, deque

import torch.nn as nn

from src.caching_helpers.base import (
    BaseCachingHelper,
    CacheAndForecastHelper,
    TaylorSeerHelper,
    HiCacheHelper,
)


class BaseCachingHelperFlux(BaseCachingHelper):
    """Base class for caching helpers of `FluxWrapper`.

    Flux calls the transformer once per sampling step, so the steps are
    counted by the transformer calls.

    Attributes:
        last_timestep: Index of the last computed step.
    """

    def wrap_transformer(self) -> None:
        """Wrap the transformer to count the steps and set `recalculate`."""
        self.function_dict['transformer_forward'] = (
            self.model.pipe.transformer.forward
        )

        def wrapped_forward(*args, **kwargs):
            self.recalculate = self.timestep not in self.ts_to_skip

            output = self.function_dict['transformer_forward'](*args, **kwargs)

            if self.recalculate:
                self.last_timestep = self.timestep
            self.timestep += 1

            return output

        self.model.pipe.transformer.forward = wrapped_forward


class FORACachingHelperFlux(BaseCachingHelperFlux):
    """FORA-style helper that reuses the heaviest layers.

    The outputs of the attention and feed-forward layers of all blocks
    are cached on computed steps and reused on skipped ones; the rest of
    the transformer is computed on every step.
    """

    def wrap_layer(self, layer: nn.Module, index: str) -> None:
        """Cache the output of the layer under the key `index`."""
        self.function_dict[index] = layer.forward

        def wrapped_forward(*args, **kwargs):
            if self.recalculate:
                self.cached_output[index] = self.function_dict[index](
                    *args, **kwargs
                )

            return self.cached_output[index]

        layer.forward = wrapped_forward

    def wrap_model(self) -> None:
        """Wrap the transformer and the cached layers of all blocks."""
        self.wrap_transformer()

        blocks = self.model.pipe.transformer.transformer_blocks

        for index, layer in enumerate(blocks):
            self.wrap_layer(layer.attn, f"double_stream attn {index}")
            self.wrap_layer(layer.ff, f"double_stream ff {index}")
            self.wrap_layer(
                layer.ff_context, f"double_stream ff_context {index}"
            )

        single_blocks = self.model.pipe.transformer.single_transformer_blocks

        for index, layer in enumerate(single_blocks):
            self.wrap_layer(layer.attn, f"single_stream attn {index}")
            self.wrap_layer(layer.proj_mlp, f"single_stream proj_mlp {index}")
            self.wrap_layer(layer.act_mlp, f"single_stream act_mlp {index}")
            self.wrap_layer(layer.proj_out, f"single_stream proj_out {index}")


class deltaDITCachingHelperFlux(BaseCachingHelperFlux):
    """Δ-DiT-style helper that reuses the delta of the single-stream blocks.

    On computed steps, the difference between the output of the last
    single-stream block and the input of the first one is cached for
    both streams. On skipped steps, the single-stream blocks are not
    computed, and the cached difference is added to their input.
    """

    def wrap_layer(self, layer: nn.Module, index: int) -> None:
        """Wrap the single-stream block with the given index."""
        self.function_dict[index] = layer.forward

        def wrapped_forward(*args, **kwargs):
            # First block: on computed steps, also save the input to
            # compute the delta.
            # Last block: on computed steps, update the delta; on skipped
            # ones, add it to the input.
            # On skipped steps, all blocks pass their input through.
            hidden_states = kwargs['hidden_states']
            encoder_hidden_states = kwargs['encoder_hidden_states']

            if index == 0 and self.recalculate:
                # save input to calculate delta
                self.cached_output['input'] = hidden_states
                self.cached_output['encoder input'] = encoder_hidden_states

            if self.recalculate:
                encoder_output, output = self.function_dict[index](
                    *args, **kwargs
                )
            else:
                encoder_output, output = encoder_hidden_states, hidden_states

            if index == self.n_end:
                if not self.recalculate:
                    encoder_output = (
                        encoder_hidden_states
                        + self.cached_output['encoder delta']
                    )
                    output = hidden_states + self.cached_output['delta']
                else:
                    self.cached_output['encoder delta'] = (
                        encoder_output - self.cached_output['encoder input']
                    )
                    self.cached_output['delta'] = (
                        output - self.cached_output['input']
                    )

            return encoder_output, output

        layer.forward = wrapped_forward

    def wrap_model(self) -> None:
        """Wrap the transformer and the single-stream blocks."""
        self.wrap_transformer()

        single_blocks = self.model.pipe.transformer.single_transformer_blocks
        self.n_end = len(single_blocks) - 1

        for index, layer in enumerate(single_blocks):
            self.wrap_layer(layer, index)


class DiCacheHelperFlux(BaseCachingHelperFlux):
    """DiCache-style helper that extrapolates the residual of all blocks.

    A residual is the difference between an output and the input of the
    first block. The first `config.probe_depth` blocks (the probe) are
    computed on every step, and the residuals of the probe and of all
    blocks are stored for the last two computed steps. On skipped steps,
    the other blocks are not computed, and the image-stream residual of
    all blocks is extrapolated as `r_2 + gamma * (r_1 - r_2)`, where
    `r_1` and `r_2` are the residuals at the last and the second-to-last
    computed steps, and `gamma = mean|p - p_2| / mean|p_1 - p_2|`,
    clamped to [1, 1.5], is computed the same way from the probe
    residuals. With only one stored residual, it is reused as is.

    Config:
        probe_depth (int): Number of blocks in the probe; must be
            positive.

    Attributes:
        residual_window: Residuals of all blocks at the last two
            computed steps.
        probe_residual_window: Residuals of the probe at the last two
            computed steps.
    """

    def wrap_layer(self, layer: nn.Module, index: int) -> None:
        """Wrap the block with the given index."""
        self.function_dict[index] = layer.forward

        def wrapped_forward(*args, **kwargs):
            encoder_hidden_states = kwargs['encoder_hidden_states']
            hidden_states = kwargs['hidden_states']

            if self.recalculate or index < self.config.probe_depth:
                encoder_output, output = self.function_dict[index](
                    *args, **kwargs
                )
            else:
                encoder_output, output = encoder_hidden_states, hidden_states

            if index == 0:
                self.cached_output['input_hidden_states'] = (
                    hidden_states.clone()
                )

            elif index == self.config.probe_depth:
                self.previous_probe_states = hidden_states.clone()

            elif index == self.n_blocks - 1:
                # here we've got the plain probe hidden_states output
                input_states = self.cached_output['input_hidden_states']

                if not self.recalculate:
                    if len(self.residual_window) >= 2:
                        p_2, p_1 = self.probe_residual_window
                        r_2, r_1 = self.residual_window

                        p = hidden_states - input_states
                        num = (p - p_2).abs().mean()
                        den = (p_1 - p_2).abs().mean().clamp_min(1e-8)
                        gamma = (num / den).clamp(1, 1.5)

                        output = input_states + r_2 + gamma * (r_1 - r_2)
                    else:
                        output = input_states + self.previous_residual

                else:
                    self.previous_residual = output - input_states
                    self.residual_window.append(self.previous_residual)

                    self.previous_probe_residual = (
                        self.previous_probe_states - input_states
                    )
                    self.probe_residual_window.append(
                        self.previous_probe_residual
                    )

            return encoder_output, output

        layer.forward = wrapped_forward

    def wrap_model(self) -> None:
        """Wrap the transformer and all blocks."""
        self.wrap_transformer()

        blocks = (
            self.model.pipe.transformer.transformer_blocks
            + self.model.pipe.transformer.single_transformer_blocks
        )
        self.n_blocks = len(blocks)

        for index, layer in enumerate(blocks):
            self.wrap_layer(layer, index)

    def set_default(self) -> None:
        """Reset the cache, the step counter and the residual windows."""
        self.recalculate = True
        self.cached_output = dict()
        self.ts_to_skip = []
        self.timestep = 0

        self.residual_window = deque(maxlen=2)
        self.probe_residual_window = deque(maxlen=2)


class CacheAndForecastHelperFlux(BaseCachingHelperFlux, CacheAndForecastHelper):
    """Base class for forecasting helpers of `FluxWrapper`.

    Forecasts the outputs of the attention and feed-forward layers of the
    double-stream blocks and of the output projection of the
    single-stream blocks. The other layers of the single-stream blocks
    are dummies: on skipped steps they are not computed, since their
    outputs only feed the forecast projection.

    Attributes:
        cache: Output and its derivative estimates at the last computed
            step, by layer key.
    """

    def wrap_layer(
        self, layer: nn.Module, index: str, dummy: bool = False
    ) -> None:
        """Wrap a layer with one output.

        Args:
            layer: Layer to wrap.
            index: Key of the layer in the cache.
            dummy: Whether the layer is a dummy: on skipped steps, it
                returns its input instead of a forecast.
        """
        self.function_dict[index] = layer.forward

        def wrapped_forward(*args, **kwargs):
            output = kwargs.get(
                'hidden_states', args if len(args) == 0 else args[0]
            )

            if self.recalculate:
                output = self.function_dict[index](*args, **kwargs)
                self.cache[index] = self.calculate_derivatives(
                    output, self.cache[index], self.last_timestep
                )

            elif not dummy:
                output = self.predict_value(
                    self.cache[index], self.last_timestep
                )

            return output

        layer.forward = wrapped_forward

    def wrap_double_output_layer(
        self, layer: nn.Module, index: str, dummy: bool = False
    ) -> None:
        """Wrap a layer that returns `(context_output, output)`.

        See `wrap_layer` for the arguments.
        """
        self.function_dict[index] = layer.forward

        def wrapped_forward(*args, **kwargs):
            hidden_states = kwargs['hidden_states']
            encoder_hidden_states = kwargs['encoder_hidden_states']

            context_output, output = encoder_hidden_states, hidden_states

            if self.recalculate:
                context_output, output = self.function_dict[index](
                    *args, **kwargs
                )

                if not dummy:
                    self.cache[(index, 'context_output')] = (
                        self.calculate_derivatives(
                            context_output,
                            self.cache[(index, 'context_output')],
                            self.last_timestep,
                        )
                    )

                    self.cache[(index, 'output')] = self.calculate_derivatives(
                        output,
                        self.cache[(index, 'output')],
                        self.last_timestep,
                    )

            else:
                if not dummy:
                    context_output = self.predict_value(
                        self.cache[(index, 'context_output')],
                        self.last_timestep,
                    )
                    output = self.predict_value(
                        self.cache[(index, 'output')], self.last_timestep
                    )

            return context_output, output

        layer.forward = wrapped_forward

    def wrap_model(self) -> None:
        """Wrap the transformer and the layers of all blocks."""
        self.wrap_transformer()

        blocks = self.model.pipe.transformer.transformer_blocks

        for index, layer in enumerate(blocks):
            self.wrap_double_output_layer(
                layer.attn, f"double_stream attn {index}"
            )
            self.wrap_layer(layer.ff, f"double_stream ff {index}")
            self.wrap_layer(
                layer.ff_context, f"double_stream ff_context {index}"
            )

        single_blocks = self.model.pipe.transformer.single_transformer_blocks

        for index, layer in enumerate(single_blocks):
            self.wrap_layer(
                layer.attn, f"single_stream attn {index}", dummy=True
            )
            self.wrap_layer(
                layer.proj_mlp, f"single_stream proj_mlp {index}", dummy=True
            )
            self.wrap_layer(
                layer.act_mlp, f"single_stream act_mlp {index}", dummy=True
            )
            self.wrap_layer(layer.proj_out, f"single_stream proj_out {index}")

    def set_default(self) -> None:
        """Reset the cache and the step counter."""
        self.recalculate = True
        self.ts_to_skip = []
        self.timestep = 0

        self.cache = defaultdict(list)
        self.last_timestep = 0


class TaylorSeerHelperFlux(CacheAndForecastHelperFlux, TaylorSeerHelper):
    """TaylorSeer helper for `FluxWrapper`."""


class HiCacheHelperFlux(CacheAndForecastHelperFlux, HiCacheHelper):
    """HiCache helper for `FluxWrapper`."""


class TaylorDPHelperFlux(TaylorSeerHelperFlux):
    """DPCache-style helper that forecasts the outputs of block stacks.

    The outputs are forecast with TaylorSeer. On skipped steps, all
    blocks pass their inputs through, except the last double-stream and
    the last single-stream blocks, whose outputs are forecast.
    """

    def wrap_model(self) -> None:
        """Wrap the transformer and all blocks."""
        self.wrap_transformer()

        layer_type = "double_stream block"
        blocks = self.model.pipe.transformer.transformer_blocks

        for index, layer in enumerate(blocks):
            dummy = index != len(blocks) - 1
            self.wrap_double_output_layer(
                layer, f"{layer_type} {index}", dummy
            )

        layer_type = "single_stream block"
        single_blocks = self.model.pipe.transformer.single_transformer_blocks

        for index, layer in enumerate(single_blocks):
            dummy = index != len(single_blocks) - 1
            self.wrap_double_output_layer(
                layer, f"{layer_type} {index}", dummy
            )


class LeMiCaHelperFlux(BaseCachingHelperFlux):
    """LeMiCa-style helper that reuses the residual of all blocks.

    On computed steps, the difference between the image-stream output of
    the last block and the input of the first one is cached. On skipped
    steps, no blocks are computed, and the cached difference is added to
    the input.
    """

    def wrap_blocks(self, layer: nn.Module, index: int) -> None:
        """Wrap the block with the given index."""
        self.function_dict[index] = layer.forward

        def wrapped_forward(*args, **kwargs):
            hidden_states = kwargs['hidden_states']
            encoder_hidden_states = kwargs['encoder_hidden_states']

            encoder_output, output = encoder_hidden_states, hidden_states

            if index == 0 and self.recalculate:
                self.cached_output['input'] = hidden_states

            if self.recalculate:
                encoder_output, output = self.function_dict[index](
                    *args, **kwargs
                )

            if index == self.n_end:
                if not self.recalculate:
                    output = hidden_states + self.cached_output['delta']
                else:
                    self.cached_output['delta'] = (
                        output - self.cached_output['input']
                    )

            return encoder_output, output

        layer.forward = wrapped_forward

    def wrap_model(self) -> None:
        """Wrap the transformer and all blocks."""
        self.wrap_transformer()

        blocks = (
            self.model.pipe.transformer.transformer_blocks
            + self.model.pipe.transformer.single_transformer_blocks
        )
        self.n_end = len(blocks) - 1

        for index, layer in enumerate(blocks):
            self.wrap_blocks(layer, index)
