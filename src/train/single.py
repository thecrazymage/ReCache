"""Single-GPU ReCache training loop for the logit predictor."""

from typing import Any

import torch
from tqdm import tqdm
from munch import Munch
from torch.utils.data import DataLoader

from src.metrics import BaseMetric
from src.caching_helpers import BaseCachingHelper
from src.logit_predictors import BaseLogitPredictor
from src.diffusion_wrappers import BaseDiffusionWrapper
from src.utils.rng import seed_everything
from src.utils.logging import BaseLogger
from src.reinforce_algorithm import sample_exp, top_k_log_prob


def train(
    args: Munch,
    model: BaseDiffusionWrapper,
    logit_predictor: BaseLogitPredictor,
    caching_helper: BaseCachingHelper,
    train_metric: BaseMetric,
    budget_scheduler: torch.Tensor,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    entropy_scheduler: torch.Tensor,
    logger: BaseLogger,
    **_: Any,
) -> None:
    """Train the logit predictor with REINFORCE on a single GPU.

    For each batch, samples `num_samples` step schedules, generates with
    caching enabled and minimizes `train_metric` with a leave-one-out baseline
    and an entropy term. Validation runs every `val_frequency` epochs,
    test every `test_frequency` epochs (both from the logger config),
    and both once more after training.

    Args:
        args: Training config, see `Config`.
        model: Diffusion wrapper.
        logit_predictor: Logit predictor to train.
        caching_helper: Caching helper that enables caching mechanism.
        train_metric: Train loss to minimize.
        budget_scheduler: Budget of each training batch, shape
            (num_batches,).
        train_dataloader: Train dataloader.
        optimizer: Optimizer over the predictor parameters.
        entropy_scheduler: Entropy coefficient for each epoch,
            shape (epochs,).
        logger: Logger, already set up.
        **_: Ignored; accepts `rank` and `world_size` of the DDP loop.

    Config:
        epochs (int): Number of training epochs.
        seed (int): Base seed; the RNGs are reseeded with `seed + epoch`
            every epoch.
        gradient_accumulation (int): Number of micro-batches per optimizer
            step, set by `main.py`; must divide `len(train_dataloader)`.
        reinforce.num_samples (int): Schedules sampled per batch;
            at least 2.
    """

    step = 0
    accumulation = 0
    global_batch_counter = 0
    num_samples = args.reinforce.num_samples
    
    assert num_samples > 1, "Num samples must be greater than 1."
    assert len(train_dataloader) % args.gradient_accumulation == 0, (
        f"len(train_dataloader)={len(train_dataloader)} must be divisible by "
        f"gradient_accumulation={args.gradient_accumulation}. "
        "Otherwise gradients leak across epochs or are dropped at the end."
    )

    # one step is an optimizer step
    for epoch in tqdm(range(args.epochs), 'epochs'):
        seed_everything(epoch + args.seed)
        if epoch % logger.config.val_frequency == 0:
            logger.log_validation(step)

        if epoch != 0 and epoch % logger.config.test_frequency == 0:
            logger.log_test(step)

        losses = 0.0
        acc_logprobs = 0.0
        acc_metrics = 0.0
        acc_metrics_reg = 0.0
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for prompts, noises, references in tqdm(
            train_dataloader, 'Training', leave=False
        ):
            if accumulation == 0:
                start_event.record()

            budget = budget_scheduler[global_batch_counter]

            global_batch_counter += 1
            accumulation += 1

            logits = logit_predictor(budget)

            noises = noises.to(model.device)
            references = references.to(model.device)

            metrics = torch.empty(
                (num_samples, len(prompts)), device=model.device
            )
            logprobs = torch.zeros(num_samples, device=model.device)

            for i in tqdm(range(num_samples), 'Reinforce steps', leave=False):
                perturbed_logits = sample_exp(logits, inference=False)

                # we have logits for every step except the first one
                ts, log_prob = top_k_log_prob(
                    logits, perturbed_logits, k=(budget - 1)
                )

                with torch.no_grad():
                    with caching_helper.inference(ts_to_skip=ts):
                        generated = model(
                            prompts=prompts, latents=noises, output_type='pt'
                        )

                logprobs[i] = log_prob

                metrics[i, :] = train_metric(
                    generated=generated,
                    references=references,
                    prompts=prompts,
                    reduction="none",
                ).detach()

            entropy_coef = entropy_scheduler[epoch]

            metrics_mean = metrics.mean(dim=0)  # without regularization
            metrics = (
                metrics + entropy_coef * logprobs[:, None]
            )  # add entropy regularizer
            metrics_mean_reg = metrics.mean(dim=0)  # [batch_size]
            metrics_corrected = (
                metrics - metrics_mean_reg[None, :]
            ).detach()  # [num_samples, batch_size]

            # n / (n - 1) converts the mean baseline to leave-one-out
            loss = (
                (metrics_corrected * logprobs[:, None]).mean()
                * (num_samples)
                / (num_samples - 1)
            ) / args.gradient_accumulation

            loss.backward()

            acc_logprobs += logprobs.detach().mean()
            acc_metrics += metrics_mean.detach().mean()
            acc_metrics_reg += metrics_mean_reg.detach().mean()
            losses += loss.detach()

            del references, generated, noises, prompts

            if accumulation % args.gradient_accumulation == 0:
                optimizer.step()

                end_event.record()
                end_event.synchronize()
                iter_time_sec = start_event.elapsed_time(end_event) / 1000.0

                logger.log_train(
                    losses,
                    logprobs=(acc_logprobs / args.gradient_accumulation),
                    before_baseline_and_reg=(
                        acc_metrics / args.gradient_accumulation
                    ),
                    before_baseline=(
                        acc_metrics_reg / args.gradient_accumulation
                    ),
                    iter_time_sec=iter_time_sec,
                    step=step,
                )

                step += 1

                accumulation = 0
                losses = 0.0
                acc_logprobs = 0.0
                acc_metrics = 0.0
                acc_metrics_reg = 0.0

                optimizer.zero_grad()

    logger.log_validation(step)
    logger.log_test(step)
