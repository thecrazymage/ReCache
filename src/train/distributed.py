"""DDP ReCache training loop for the logit predictor."""

from contextlib import nullcontext

import torch
from tqdm import tqdm
from munch import Munch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils.rng import seed_everything
from src.utils.ddp import ddp_reduce_mean
from src.reinforce_algorithm import sample_exp, top_k_log_prob
from src.logit_predictors import BaseLogitPredictor
from src.metrics import BaseMetric
from src.caching_helpers import BaseCachingHelper
from src.diffusion_wrappers import BaseDiffusionWrapper
from src.utils.logging import BaseLogger


def train(
    args: Munch,
    model: BaseDiffusionWrapper,
    logit_predictor: BaseLogitPredictor | DDP,
    caching_helper: BaseCachingHelper,
    train_metric: BaseMetric,
    budget_scheduler: torch.Tensor,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    entropy_scheduler: torch.Tensor,
    logger: BaseLogger,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """DDP version of `src.train.single.train`.

    Must be called on all ranks. Each rank trains on its own shard with
    micro-batches of size 1 and reseeds the RNGs with
    `seed + epoch + 10000 * rank` every epoch. Gradients are synchronized
    only on the last micro-batch of each accumulation, and logged train
    values are averaged over ranks.

    Args:
        logit_predictor: Logit predictor to train, possibly wrapped in DDP.
        train_dataloader: Train dataloader with a `DistributedSampler`.
        rank: Rank of the current process.
        world_size: Number of ranks.

        See `src.train.single.train` for the other arguments and config.
    """
    is_main_process = rank == 0

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

    for epoch in tqdm(
        range(args.epochs), 'Epochs', disable=not is_main_process
    ):
        train_dataloader.sampler.set_epoch(epoch)

        seed_everything(epoch + args.seed + 10000 * rank)

        if epoch % logger.config.val_frequency == 0:
            logger.log_validation(step)

        if epoch != 0 and epoch % logger.config.test_frequency == 0:
            logger.log_test(step)

        losses = torch.tensor(0.0, device=model.device)
        acc_logprobs = torch.tensor(0.0, device=model.device)
        acc_metrics = torch.tensor(0.0, device=model.device)
        acc_metrics_reg = torch.tensor(0.0, device=model.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for prompts, noises, references in train_dataloader:
            assert len(prompts) == 1
            if accumulation == 0:
                start_event.record()

            budget = budget_scheduler[global_batch_counter]

            global_batch_counter += 1
            accumulation += 1

            sync_gradients = accumulation % args.gradient_accumulation == 0
            sync_context = (
                logit_predictor.no_sync()
                if isinstance(logit_predictor, DDP) and not sync_gradients
                else nullcontext()
            )

            with sync_context:
                logits = logit_predictor(budget)

                noises = noises.to(model.device)
                references = references.to(model.device)

                metrics = torch.empty(
                    (num_samples, len(prompts)), device=model.device
                )
                logprobs = torch.zeros(num_samples, device=model.device)

                for i in range(num_samples):
                    perturbed_logits = sample_exp(logits, inference=False)
                    ts, log_prob = top_k_log_prob(
                        logits, perturbed_logits, k=(budget - 1)
                    )

                    with torch.inference_mode():
                        with caching_helper.inference(ts_to_skip=ts):
                            generated = model(
                                prompts=prompts,
                                latents=noises,
                                output_type='pt',
                            )

                    logprobs[i] = log_prob

                    metrics[i, :] = train_metric(
                        generated=generated,
                        references=references,
                        prompts=prompts,
                        reduction="none",
                    ).detach()

                entropy_coef = entropy_scheduler[epoch]
                metrics_mean = metrics.mean(dim=0)
                metrics = metrics + entropy_coef * logprobs[:, None]

                metrics_mean_reg = metrics.mean(dim=0)
                metrics_corrected = (
                    metrics - metrics_mean_reg[None, :]
                ).detach()

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

            if sync_gradients:
                optimizer.step()
                end_event.record()
                end_event.synchronize()
                iter_time_sec = start_event.elapsed_time(end_event) / 1000.0

                losses_to_log = ddp_reduce_mean(losses, world_size)
                acc_logprobs = acc_logprobs / args.gradient_accumulation
                acc_metrics = acc_metrics / args.gradient_accumulation
                acc_metrics_reg = acc_metrics_reg / args.gradient_accumulation

                acc_logprobs_to_log = ddp_reduce_mean(acc_logprobs, world_size)
                acc_metrics_to_log = ddp_reduce_mean(acc_metrics, world_size)
                acc_metrics_reg_to_log = ddp_reduce_mean(
                    acc_metrics_reg, world_size
                )

                if is_main_process:
                    logger.log_train(
                        losses_to_log,
                        logprobs=acc_logprobs_to_log,
                        before_baseline_and_reg=acc_metrics_to_log,
                        before_baseline=acc_metrics_reg_to_log,
                        iter_time_sec=iter_time_sec,
                        step=step,
                    )

                step += 1

                accumulation = 0
                losses = torch.tensor(0.0, device=model.device)
                acc_logprobs = torch.tensor(0.0, device=model.device)
                acc_metrics = torch.tensor(0.0, device=model.device)
                acc_metrics_reg = torch.tensor(0.0, device=model.device)

                optimizer.zero_grad(set_to_none=True)

    logger.log_validation(step)
    logger.log_test(step)
