"""Evaluate a caching baseline with uniform schedules, or the teacher.

Test metrics are logged to Comet for each number of computed steps in
`logging.parameters.steps`, see `BaselinesLogger`.
"""

import click

from src.utils.models import get_models
from src.utils.logging import get_logger
from src.utils.data.setup import get_dataloaders
from src.utils.config import config_from_path
from src.utils.rng import seed_everything
from src.diffusion_wrappers import get_diffusion_wrapper


@click.command()
@click.option(
    "--config", type=str, required=True, help="Path to YAML config file"
)
@click.option(
    "--reference",
    is_flag=True,
    help="Evaluate the full-NFE teacher instead of the baseline",
)
@click.option(
    "--run_name",
    type=str,
    default="recache training",
    help="Run name for logger",
)
@click.option("--device", type=str, default="cuda:0", help="Device to run on")
@click.option("--seed", type=int, default=42, help="Random seed")
def main(
    config: str, reference: bool, run_name: str, device: str, seed: int
) -> None:
    """Evaluate a caching baseline or, with --reference, the teacher."""
    args = config_from_path(config)

    args.seed = seed
    seed_everything(seed)

    logger = get_logger(args, run_name)

    if not reference:
        model, caching_helper = get_models(
            args.diffusion_wrapper, args.caching_helper, device=device
        )
    else:
        model = get_diffusion_wrapper(
            args.diffusion_wrapper.wrapper_class,
            args.diffusion_wrapper.parameters,
            device,
        )

        caching_helper = None

    args.dataset.train_size = 0
    args.dataset.val_size = 0

    test_dataloader, db_dataloader = get_dataloaders(
        args.dataset, args.dataset.batch_size
    )[-2:]

    logger.setup(model, caching_helper, test_dataloader, db_dataloader)

    if not reference:
        logger.log_multistep()
    else:
        logger.log_teacher()

    logger.finalize()


if __name__ == '__main__':
    main()
