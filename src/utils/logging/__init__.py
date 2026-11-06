"""Training loggers and the factory that builds them from a config."""

from munch import Munch

from src.utils.logging.base import BaseLogger
from src.utils.logging.comet import CometLogger
from src.utils.logging.baselines import BaselinesLogger


def get_logger(config: Munch, run_name: str, rank: int = 0) -> BaseLogger:
    """Build the logger named in `config.logging.logger_class`.

    Args:
        config: Full run config; the logger reads its own section.
        run_name: Run name shown in the experiment tracker.
        rank: Process rank. Ignored by `BaselinesLogger`.

    Raises:
        ValueError: If the logger class is unknown.
    """
    
    logger_class = config.logging.logger_class

    if logger_class == "CometLogger":
        return CometLogger(config, run_name, rank)

    if logger_class == "BaselinesLogger":
        return BaselinesLogger(config, run_name)

    raise ValueError(f"unknown logger class: {logger_class}")
