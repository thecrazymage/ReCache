"""Sampling of step schedules and their log-probabilities for REINFORCE.

A schedule containing `k` of `N` steps is drawn from the Plackett-Luce distribution: 
it outputs an ordering (i.e. importance) of steps `1, ..., N`, 
first `k` (the most important) of which are computed at inference. 
The weights are proportional to `exp(-logits)`; step 0 is always computed. 

In training, an ordering is sampled with the Exp-Min (Gumbel-Max) trick, 
its first `k` steps are computed and the rest are skipped. 

At inference, there is no noise: the steps are sorted by 
their logits, which gives the most probable ordering, 
and the `k` steps with the lowest logits are computed.
"""

import torch


def sample_exp(logits: torch.Tensor, inference: bool = False) -> torch.Tensor:
    """Perturb logits with the Exp-Min trick.

    Returns `exp(l_i + g_i)`, where `l_i` is the i-th logit and
    `g_i = log(-log(U_i))`, `U_i ~ Uniform(0, 1)`, is negated standard 
    Gumbel noise. The resulting samples follow exponential distribution 
    `Exp(exp(-l_i))`. Sorting these sample in ascending order is equivalent 
    to drawing the steps one by one without replacement, each time 
    picking a remaining step with probability proportional to `exp(-l_i)`.

    Args:
        logits: Logits of shape (N,).
        inference: Whether to return `exp(logits)` without noise. Its
            ascending order is the most probable ordering; the
            `budget - 1` steps with the lowest values are computed.

    Returns:
        Positive values of shape (N,), to be sorted in ascending order.
    """
    if inference:
        return torch.exp(logits)

    uniform_noise = torch.distributions.utils.clamp_probs(
        torch.rand_like(logits)
    )
    exp_noise = torch.log(-torch.log(uniform_noise))

    sampled_values = torch.exp(logits + exp_noise)

    return sampled_values


def top_k_log_prob(
    logits: torch.Tensor,
    perturbed_logits: torch.Tensor,
    k: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the log-probability of the first `k` sampled steps.

    The sampled ordering `t` sorts `perturbed_logits` in ascending order.
    With `θ = logits`, the probability of its first `k` elements is

        P(t_1, ..., t_k) = ∏_{i=1..k} [ exp(-θ_{t_i}) /
            Σ_{j ∉ {t_1, ..., t_{i-1}}} exp(-θ_j) ],

        log P(t_1, ..., t_k) = Σ_{i=1..k} [ -θ_{t_i} -
            log Σ_{j ∉ {t_1, ..., t_{i-1}}} exp(-θ_j) ].

    The lower the logit, the more important the step.

    Args:
        logits: Logits of shape (N,) for steps 1, ..., N.
        perturbed_logits: Output of `sample_exp` for `logits`, of shape
            (N,).
        k: Number of computed steps, excluding step 0.

    Returns:
        Tuple `(ts_to_skip, log_prob)`: indices of the N - k skipped
        steps, shifted by 1 to account for step 0, and the scalar
        `log P(t_1, ..., t_k)`.
    """
    num_total_steps = logits.shape[0]  # (N, )

    # The ordered sequence t = (t_1, ..., t_k), equivalent to taking
    # argmin k times.
    indices = torch.argsort(perturbed_logits)
    top_k_indices = indices[:k]  # (k, )

    # Binary mask with bin_mask[i, j] = 1 if t_i = j
    all_indices = torch.arange(num_total_steps, device=logits.device)
    bin_mask = top_k_indices[:, None] == all_indices[None, :]  # (k, N)

    # Numerator: pick the logit of t_i for each step i
    log_numerator = -(logits[None, :] * bin_mask).sum(dim=-1)  # (k,)

    # Denominator
    # Mask that excludes the already chosen t_1, ..., t_{i-1} at step i
    exclusion_mask = torch.cumsum(bin_mask, dim=0)[:-1, :]  # (k-1, N)
    # Add a row of zeros: nothing is excluded at step 1
    exclusion_mask = torch.cat(
        [
            torch.zeros(1, num_total_steps, device=logits.device),
            exclusion_mask,
        ],
        dim=0,
    )  # (k, N)
    # Apply the mask: set the already chosen logits to -inf, so that they
    # do not contribute to logsumexp
    exclusion_mask[exclusion_mask > 0] = float('inf')
    masked_logits = -logits[None, :] - exclusion_mask
    log_denominator = torch.logsumexp(masked_logits, dim=-1)  # (k,)

    # Put it all together
    log_prob = (log_numerator - log_denominator).sum()

    # The caching helper takes the steps to skip; +1 accounts for step 0
    return indices[k:] + 1, log_prob
