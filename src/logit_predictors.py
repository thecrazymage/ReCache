"""Logit predictors that score diffusion steps for caching.

A predictor outputs one logit per step except the first one, which is
always computed: logit `i` belongs to step `i + 1`. Lower logits make a
step more likely to be computed; see `src.reinforce_algorithm` for how
schedules are sampled from the logits. The logits may depend on the
budget, the number of fully computed steps including the first one.
"""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from munch import Munch


class BaseLogitPredictor(ABC, nn.Module):
    """Base class for logit predictors.

    Args:
        n_logits: Number of logits, i.e. the number of diffusion steps
            minus one.
        config: Predictor config, see the subclasses.
        dtype: Dtype of the predictor parameters.
        device: Device to put the predictor on.
    """

    def __init__(
        self,
        n_logits: int,
        config: Munch | None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        super().__init__()

        self.config = config
        self.dtype = dtype
        self.device = device

        self.n_logits = n_logits

    @abstractmethod
    def model_parameters(self) -> list[nn.Parameter]:
        """Return the parameters to optimize."""
        pass

    @abstractmethod
    def forward(self, budget: torch.Tensor) -> torch.Tensor:
        """Predict the step logits.

        Args:
            budget: Caching budget as a 0-d tensor.

        Returns:
            Logits of shape (n_logits,).
        """
        pass


class MLP(BaseLogitPredictor):
    """MLP predictor, optionally conditioned on the budget.

    With budget conditioning, the MLP input is a sinusoidal embedding of
    the budget. Otherwise it is a fixed random vector, so the logits do
    not depend on it.

    See `BaseLogitPredictor` for the arguments.

    Config:
        input_dim (int): Dimension of the MLP input.
        embed_dims (list[int]): Widths of the hidden ReLU layers; `[]`
            gives a single linear layer.
        budget_conditioning (bool): Whether to condition on the budget.
        budget_embedding_interpolate (bool): Whether to rescale the budget
            from [0, n_logits + 1] to [0, budget_embedding_max_period]
            before embedding it. Used only with budget conditioning.
        budget_embedding_max_period (int): Max period of the sinusoidal
            embedding. Used only with budget conditioning.
    """

    def __init__(
        self,
        n_logits: int,
        config: Munch,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        super().__init__(n_logits, config, dtype, device)

        self.budget_cond_enabled = self.config.budget_conditioning

        self.predictor = self.init_predictor()

        self.register_buffer(
            'constant_input',
            torch.randn((1, self.config.input_dim), dtype=self.dtype),
        )

        self.to(self.device, dtype=self.dtype)

    def init_predictor(self) -> nn.Sequential:
        """Build the MLP from `input_dim`, `embed_dims` and `n_logits`."""
        layers = []

        output_dim = self.config.input_dim
        for dim in self.config.embed_dims:
            input_dim = output_dim
            output_dim = dim

            layers += [nn.Linear(input_dim, dim), nn.ReLU()]
        layers += [nn.Linear(output_dim, self.n_logits)]

        return nn.Sequential(*layers)

    def model_parameters(self) -> list[nn.Parameter]:
        return list(self.predictor.parameters())

    def forward(self, budget: torch.Tensor) -> torch.Tensor:
        if self.budget_cond_enabled:
            budget = budget.unsqueeze(0).to(self.device)
            model_input = self.get_budget_embedding(budget).to(
                dtype=self.dtype
            )
        else:
            model_input = self.constant_input

        logits = self.predictor(model_input).squeeze(0)

        return logits

    def get_budget_embedding(self, budget: torch.Tensor) -> torch.Tensor:
        """Embed budgets with sinusoids.

        Adapted from the DDPM implementation in diffusers.

        Args:
            budget: Budgets of shape (N,).

        Returns:
            Float32 embeddings of shape (N, input_dim).
        """

        assert len(budget.shape) == 1, "budget should be a 1d-array"

        if self.config.budget_embedding_interpolate:
            budget = (
                (self.config.budget_embedding_max_period * budget)
                / (self.n_logits + 1)
            ).long()

        half_dim = self.config.input_dim // 2
        exponent = -math.log(
            self.config.budget_embedding_max_period
        ) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=budget.device
        )
        exponent = exponent / (half_dim - 1)
        emb = torch.exp(exponent)
        emb = budget[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.config.input_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

        return emb


class LogitsVector(BaseLogitPredictor):
    """Learnable logits that do not depend on the budget.

    The logits are initialized from the standard normal distribution.
    `config` is unused. See `BaseLogitPredictor` for the arguments.
    """

    def __init__(
        self,
        n_logits: int,
        config: Munch | None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        super().__init__(n_logits, config, dtype, device)

        self.predictor = torch.nn.Parameter(
            torch.randn(n_logits, dtype=self.dtype), requires_grad=True
        )
        self.to(self.device)

    def model_parameters(self) -> list[nn.Parameter]:
        return [self.predictor]

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        return self.predictor


def load_logit_model(
    n_logits: int,
    config: Munch,
    dtype: torch.dtype,
    device: torch.device | str = torch.device('cuda'),
) -> BaseLogitPredictor:
    """Build the logit predictor selected in the config.

    See `BaseLogitPredictor` for the arguments.

    Config:
        predictor_variant (str): `'MLP'` or `'LogitsVector'`.
        parameters (Munch | None): Config of the predictor class.

    Raises:
        NotImplementedError: If `predictor_variant` is unknown.
    """

    if config.predictor_variant == 'MLP':
        return MLP(n_logits, config.parameters, dtype, device)

    if config.predictor_variant == 'LogitsVector':
        return LogitsVector(n_logits, config.parameters, dtype, device)

    raise NotImplementedError(
        f'unknown logit model passed, {config.predictor_variant}'
    )
