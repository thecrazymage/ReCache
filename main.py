"""Train a logit predictor that selects which diffusion steps to cache.

Runs on a single GPU, or with DDP when launched via `torchrun`.
Example configs are in `configs/<model>/`.
"""
import os
import logging

import torch
import click

from src.train import single_train, distributed_train
from src.utils.models import get_models
from src.logit_predictors import load_logit_model
from src.metrics import metrics_registry
from src.utils.logging import get_logger
from src.utils.data.setup import get_dataloaders
from src.utils.config import config_from_path
from src.utils.ddp import ddp_setup, ddp_end
from src.utils.rng import seed_everything, weighted_randint


@click.command()
@click.option("--config", type=str, required=True, help="Path to YAML config file")
@click.option("--run_name", type=str, default="recache training", help="Run name for logger")
@click.option("--device", type=str, default="cuda:0", help="Device for single-GPU runs, ignored under torchrun")
@click.option("--seed", type=int, default=42, help="Random seed, overrides training.seed from the config")
def main(config: str, run_name: str, device: str, seed: int) -> None:
    """Train a logit predictor for diffusion caching."""
    args = config_from_path(config)
    args.training.seed = seed

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        local_rank, rank, world_size = ddp_setup()
        device = f"cuda:{local_rank}"
    else:
        torch.cuda.set_device(device)
        rank, world_size = 0, 1

    if distributed and rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    seed_everything(seed + rank)

    logger = get_logger(args, run_name, rank)
    model, caching_helper = get_models(
        args.diffusion_wrapper, args.caching_helper, device=device
    )

    n_logits = model.config.nfe - 1
    logit_predictor = load_logit_model(
        n_logits=n_logits,
        config=args.logit_predictor,
        dtype=torch.float32,
        device=device,
    )
    if distributed:
        train_logit_predictor = torch.nn.parallel.DistributedDataParallel(
            logit_predictor,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    else:
        train_logit_predictor = logit_predictor

    args = args.training.copy()
    assert args.global_batch_size % (args.micro_batch_size * world_size) == 0
    args.gradient_accumulation = args.global_batch_size // (
        args.micro_batch_size * world_size
    )

    train_metric = metrics_registry[args.metric.metric_name](
        config=args.metric.parameters,
        is_train=True,
        is_video=model.is_video,
        device=device,
    )

    optimizer = torch.optim.Adam(
        logit_predictor.model_parameters(), lr=args.optimizer_lr
    )

    train_dataloader, val_dataloader, test_dataloader, db_dataloader = (
        get_dataloaders(
            args.dataset, args.micro_batch_size, distributed, world_size, rank
        )
    )

    total_batches = args.epochs * len(train_dataloader)
    budget_scheduler = weighted_randint(total_batches, args.budget_scheduler)
    entropy_scheduler = torch.linspace(
        args.entropy_alpha, 0.0, args.epochs, device=device
    )

    logger.setup(
        model,
        logit_predictor,
        caching_helper,
        train_metric,
        optimizer,
        budget_scheduler,
        val_dataloader,
        test_dataloader,
        db_dataloader,
    )

    if rank == 0:
        print("\nStarting training!\n")

    train = distributed_train if distributed else single_train

    train(
        args,
        model,
        train_logit_predictor,
        caching_helper,
        train_metric,
        budget_scheduler,
        train_dataloader,
        optimizer,
        entropy_scheduler,
        logger,
        rank=rank,
        world_size=world_size,
    )

    if rank == 0:
        print("\nTraining completed successfully!\n")
        logger.finalize()

    if distributed:
        torch.distributed.barrier()

    logger.log_learned_multistep()

    if distributed:
        ddp_end()


if __name__ == '__main__':
    main()
