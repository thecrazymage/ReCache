"""Base logger: run directory, sample generation and checkpointing."""

import os
from collections import deque
from collections.abc import Sequence
from datetime import datetime

import yaml
import torch
from munch import Munch
from tqdm import tqdm
from diffusers.utils import export_to_video
from torch.utils.data import DataLoader

from src.metrics import BaseMetric
from src.reinforce_algorithm import sample_exp
from src.caching_helpers import BaseCachingHelper
from src.logit_predictors import BaseLogitPredictor
from src.diffusion_wrappers import BaseDiffusionWrapper
from src.utils.logging.utils import to_video_frames


class BaseLogger:
    """Base class for training loggers.

    Creates the run directory, generates samples with the learned or a
    given step schedule and saves checkpoints.

    Under DDP, every rank generates samples and computes metrics on its
    own data shard, but only rank 0 (`is_main_process`) writes files and
    logs to the tracker.

    Args:
        config: Full run config; logger settings are read from
            `config.logging.parameters`.
        run_name: Run name, also used in the run directory name.
        rank: Process rank.

    Config:
        eval_budget (int): Budget, i.e. the number of fully computed
            steps including the first one, for validation and test.
        checkpoint_path (str): Root directory for runs. Logs are saved
            to `<checkpoint_path>/<run_name>_<timestamp>`, and the config
            value is updated in place to point there.
        grad_mean_num_iters (int): Window of the predictor grad norm
            moving average.
    """

    def __init__(self, config: Munch, run_name: str, rank: int = 0) -> None:
        self.run_name = run_name
        self.is_main_process = rank == 0

        self.config = config.logging.parameters
        self.eval_budget = self.config.eval_budget

        self.run_config = config.toDict()

        if self.is_main_process:
            checkpoint_dir = f"{self.run_name}_{datetime.now():%Y%m%d_%H%M%S}"
            self.config.checkpoint_path = os.path.join(
                self.config.checkpoint_path, checkpoint_dir
            )

            self.run_config["logging"]["parameters"][
                "checkpoint_path"
            ] = self.config.checkpoint_path
            
            os.makedirs(self.config.checkpoint_path, exist_ok=True)

            config_path = os.path.join(
                self.config.checkpoint_path, "config.yaml"
            )
            with open(config_path, "w") as f:
                yaml.safe_dump(
                    config.toDict(), f, sort_keys=False, allow_unicode=True
                )

            print(
                f"""
                🍄 Logger initialized!
                Run name: {self.run_name}
                Checkpoint path: {self.config.checkpoint_path}
            """
            )

    def setup(
        self,
        model: BaseDiffusionWrapper,
        logit_predictor: BaseLogitPredictor,
        caching_helper: BaseCachingHelper,
        train_metric: BaseMetric,
        optimizer: torch.optim.Optimizer,
        budget_scheduler: torch.Tensor,
        val_dataloader: DataLoader | None,
        test_dataloader: DataLoader | None,
        db_dataloader: DataLoader | None,
    ) -> None:
        """Setup the components used for logging.

        Dataloaders are None for disabled splits. Also logs the
        `budget_scheduler` distribution.
        """
        self.model = model
        self.logit_predictor = logit_predictor
        self.caching_helper = caching_helper
        self.train_metric = train_metric
        self.optimizer = optimizer

        self.logit_predictor_grad_mean = deque(
            maxlen=self.config.grad_mean_num_iters
        )

        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.db_dataloader = db_dataloader

        if self.is_main_process:
            self.log_budget_scheduler(budget_scheduler)

    def log_budget_scheduler(self, budget_scheduler: torch.Tensor) -> None:
        """Log the distribution of per-batch budgets.

        Args:
            budget_scheduler: Budget of each training batch, shape
                (num_batches,).
        """
        raise NotImplementedError

    def log_train(
        self,
        loss: torch.Tensor,
        logprobs: torch.Tensor,
        before_baseline_and_reg: torch.Tensor,
        before_baseline: torch.Tensor,
        iter_time_sec: float,
        step: int,
    ) -> None:
        """Log metrics of one optimizer step.

        Values are averaged over gradient accumulation steps and, under
        DDP, over ranks.

        Args:
            loss: REINFORCE loss.
            logprobs: Mean log-probability of the sampled step sets.
            before_baseline_and_reg: Mean training metric of the samples.
            before_baseline: Mean training metric plus the entropy term,
                before the baseline is subtracted.
            iter_time_sec: Wall time of the optimizer step in seconds.
            step: Optimizer step index.
        """
        raise NotImplementedError

    def log_validation(self, step: int) -> None:
        """Evaluate the learned schedule on the validation set."""
        raise NotImplementedError

    def log_test(self, step: int) -> None:
        """Evaluate the learned schedule on the test set."""
        raise NotImplementedError

    def finalize(self) -> None:
        """Close the tracker at the end of the run; no-op by default."""
        pass

    def generate_samples(
        self,
        dataloader: DataLoader,
        budget: int | None = None,
        ts_to_skip: torch.Tensor | Sequence[int] | None = None,
        description: str = '',
        output_video: bool = False,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | Sequence[int],
        torch.Tensor,
        list[str],
        torch.Tensor,
    ]:
        """Generate samples with a given or the learned step schedule.

        If `ts_to_skip` is not given, uses the deterministic schedule of
        the logit predictor for `budget`.

        Args:
            dataloader: Dataloader yielding `(prompts, noises, references)`.
            budget: Number of fully computed steps, including the first
                one. Required if `ts_to_skip` is None.
            ts_to_skip: Indices of steps to skip.
            description: Progress bar label.
            output_video: Whether to return full videos for video models.

        Returns:
            Tuple `(logits, ts_to_skip, generated, prompts, references)`.
            `logits` are the predictor outputs, or None if `ts_to_skip`
            was given. `generated` and `references` are stacked on CPU.
        """
        logits = None

        if ts_to_skip is None:
            assert budget is not None, "you should specify the budget"

            logits = self.logit_predictor(torch.tensor(budget))

            mode_logits = sample_exp(logits, inference=True)
            ts_to_skip = torch.argsort(mode_logits)[budget - 1:] + 1

        generated, prompts, references = [], [], []

        for batch in tqdm(
            dataloader,
            desc=description,
            leave=False,
            disable=not self.is_main_process,
        ):
            cur_prompts, cur_noises = batch[:2]
            cur_noises = cur_noises.to(self.model.device)

            with self.caching_helper.inference(ts_to_skip=ts_to_skip):
                samples = self.model(
                    prompts=cur_prompts,
                    latents=cur_noises,
                    output_type='pt',
                    output_video=output_video,
                ).cpu()

            prompts += cur_prompts
            generated.append(samples)

            references.append(batch[2])

        generated = (
            torch.concat(generated, dim=0) if generated else torch.empty(0)
        )
        references = (
            torch.concat(references, dim=0) if references else torch.empty(0)
        )

        return logits, ts_to_skip, generated, prompts, references

    def save_checkpoint(
        self, step: int, video_tensor: torch.Tensor | None = None
    ) -> None:
        """Save predictor and optimizer states for a training step.

        Writes `<checkpoint_path>/<step>/model.pt`.

        Args:
            step: Training step, used as the subdirectory name.
            video_tensor: Videos to save alongside, see `save_video`.
        """
        save = {
            'logit_model': self.logit_predictor.state_dict(),
            'optim': self.optimizer.state_dict(),
        }
        path = os.path.join(self.config.checkpoint_path, str(step))
        checkpoint_path = os.path.join(path, 'model.pt')

        os.makedirs(path, exist_ok=True)
        torch.save(save, checkpoint_path)

        print(f'checkpoint saved in {checkpoint_path}')

        if video_tensor is not None:
            self.save_video(step, video_tensor)

    def save_video(self, step: int | str, video_tensor: torch.Tensor) -> None:
        """Save videos as `<checkpoint_path>/<step>/videos/<i>.mp4`.

        Args:
            step: Subdirectory name, usually the training step.
            video_tensor: Videos of shape (N, T, C, H, W).
        """
        path = os.path.join(self.config.checkpoint_path, str(step))

        video_path = os.path.join(path, 'videos')
        os.makedirs(video_path, exist_ok=True)

        for i in range(len(video_tensor)):
            current_path = os.path.join(video_path, f'{i}.mp4')
            export_to_video(
                to_video_frames(video_tensor[i]),
                current_path,
                fps=self.model.config.fps,
            )

        print(f'videos saved in {video_path}')
