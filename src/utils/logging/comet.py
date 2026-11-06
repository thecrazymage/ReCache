"""Comet logger for train runs."""

import os

import comet_ml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from munch import Munch

from src.metrics import metrics_registry
from src.utils.logging.utils import (
    prepare_images,
    draw_timesteps_plot,
    calc_metrics,
)
from src.utils.ddp import ddp_reduce_sum
from src.utils.logging.base import BaseLogger


class CometLogger(BaseLogger):
    """Logger that reports training progress to Comet.

    Args:
        config: Full run config, see `BaseLogger`.
        run_name: Comet experiment name.
        rank: Process rank.

    Config:
        project_name (str): Comet project.
        batch_size (int): Batch size for metric computation.
        log_size (int): Number of samples in logged image grids and
            saved videos.
        test_metrics (list[str]): Test metric names in
            `metrics_registry`.
        log_multistep.project_name (str): Comet project for
            `log_learned_multistep`.
        log_multistep.budgets (list[int]): Evaluation 
            budgets for `log_learned_multistep`.

        See `BaseLogger` for the remaining keys.

    """

    def __init__(self, config: Munch, run_name: str, rank: int = 0) -> None:
        super().__init__(config, run_name, rank)

        if self.is_main_process:
            self.experiment = comet_ml.start(
                project_name=self.config.project_name
            )

            self.experiment.set_name(run_name)
            self.experiment.log_parameters(
                self.run_config, nested_support=True
            )

            # Code logging
            for path in os.listdir('./'):
                if path.endswith('.py'):
                    self.experiment.log_code(file_name=path)
            if os.path.isdir('./src'):
                self.experiment.log_code(folder='./src')

    def log_budget_scheduler(self, budget_scheduler: torch.Tensor) -> None:
        budgets = budget_scheduler.detach().to(torch.long).cpu()
        min_budget = int(budgets.min().item())
        max_budget = int(budgets.max().item())
        bins = range(min_budget, max_budget + 2)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(budgets.numpy(), bins=bins, align='left', rwidth=0.8)
        ax.set_title('Distribution of sampled budgets')
        ax.set_xlabel('Budget')
        ax.set_ylabel('Frequency')
        ax.set_xticks(range(min_budget, max_budget + 1))
        ax.grid(axis='y', alpha=0.75)

        self.experiment.log_figure(
            figure=fig, figure_name='budget_scheduler_hist'
        )
        plt.close(fig)

    def log_train(
        self,
        loss: torch.Tensor,
        logprobs: torch.Tensor,
        before_baseline_and_reg: torch.Tensor,
        before_baseline: torch.Tensor,
        iter_time_sec: float,
        step: int,
    ) -> None:
        """Log train metrics, the predictor grad norm and its average."""

        predictor_parameters = [
            p
            for p in self.logit_predictor.model_parameters()
            if p.grad is not None and p.requires_grad
        ]
        predictor_grad_norm = 0
        for p in predictor_parameters:
            predictor_grad_norm += p.grad.norm(2).item() ** 2
        predictor_grad_norm = predictor_grad_norm**0.5
        self.logit_predictor_grad_mean.append(predictor_grad_norm)

        log_dict = {
            'train/train loss': loss.item(),
            (
                f'train/train {str(self.train_metric)} '
                'before baseline and regularization'
            ): before_baseline_and_reg.item(),
            f'train/train {str(self.train_metric)} before baseline': (
                before_baseline.item()
            ),
            'train/log prob': logprobs.item(),
            'train/logit predictor grad norm': predictor_grad_norm,
            'train/logit predictor grad norm moving average': float(
                np.nanmean(self.logit_predictor_grad_mean)
            ),
            'train/one iteration time (in seconds)': iter_time_sec,
        }

        self.experiment.log_metrics(log_dict, step=step)

    @torch.no_grad()
    def log_validation(self, step: int) -> None:
        """Evaluate the learned schedule on the validation set.

        Uses budget `eval_budget`. Logs the training metric, the
        predictor logits, a sample grid and the schedule plot. Must be
        called on all ranks.
        """
        if self.val_dataloader is None:
            return

        self.logit_predictor.eval()

        logits, ts_to_skip, generated, prompts, references = (
            self.generate_samples(
                self.val_dataloader,
                budget=self.eval_budget,
                description='validation',
            )
        )

        self.logit_predictor.train()

        metric = self.train_metric.calculate_dataset(
            generated=generated,
            references=references,
            prompts=prompts,
            batch_size=self.config.batch_size,
            reduction='sum',
        )
        sz = torch.tensor(len(prompts), device=metric.device)
        metric = ddp_reduce_sum(metric) / ddp_reduce_sum(sz)

        if not self.is_main_process:
            return

        log_data = {
            f'val/{str(self.train_metric)}': metric,
            **{
                f'val/logits/logit_{i + 1}': v.item()
                for i, v in enumerate(logits)
            },
        }

        self.experiment.log_metrics(log_data, step=step)

        grid = prepare_images(
            references[-self.config.log_size:],
            generated[-self.config.log_size:],
        )
        self.experiment.log_image(
            image_data=grid, name='val/samples', step=step
        )

        fig = draw_timesteps_plot(ts_to_skip, self.model.config.nfe, step)
        self.experiment.log_figure(
            figure=fig, figure_name='val/timesteps_plot', step=step
        )
        plt.close(fig)

    @torch.no_grad()
    def log_test(self, step: int, multistep: bool = False) -> None:
        """Evaluate the learned schedule on enabled test and DrawBench splits.

        Logs test metrics, a sample grid and the schedule plot. 
        Outside multistep mode, also saves a checkpoint, with sample videos 
        for video models. Must be called on all ranks, 
        since metrics are all-reduced.

        Args:
            step: Training step. In multistep mode, is the evaluated budget.
            multistep: Whether this is a `log_learned_multistep` call;
                otherwise the budget is `eval_budget`.
        """
        if self.is_main_process and not multistep:
            self.save_checkpoint(step)

        budget = step if multistep else self.eval_budget

        if self.test_dataloader is None:
            self._log_drawbench(step, budget)
            return

        self.logit_predictor.eval()

        _, ts_to_skip, generated, prompts, references = self.generate_samples(
            self.test_dataloader,
            budget=budget,
            description='test',
            output_video=self.model.is_video,
        )

        self.logit_predictor.train()

        if self.is_main_process:
            video_tensor = (
                generated[-self.config.log_size:]
                if self.model.is_video
                else None
            )

            if not multistep and video_tensor is not None:
                self.save_video(step, video_tensor)

            if self.model.is_video:
                grid = prepare_images(
                    references[-self.config.log_size:, 0],
                    generated[-self.config.log_size:, 0],
                )
            else:
                grid = prepare_images(
                    references[-self.config.log_size:],
                    generated[-self.config.log_size:],
                )
            self.experiment.log_image(
                image_data=grid, name='test/samples', step=step
            )

            fig = draw_timesteps_plot(ts_to_skip, self.model.config.nfe, step)
            self.experiment.log_figure(
                figure=fig, figure_name='test/timesteps_plot', step=step
            )
            plt.close(fig)


        with torch.inference_mode():
            metrics = calc_metrics(
                metric_names=self.config.test_metrics,
                device=self.model.device,
                generated=generated,
                references=references,
                prompts=prompts,
                batch_size=self.config.batch_size,
                is_video=self.model.is_video,
            )

        if self.is_main_process:
            print(
                f'\n\n🏔  test metrics for budget {budget} calculated!\n'
            )
            for name, value in metrics.items():
                print(name.split('/')[-1], value)
            print('\n')

            self.experiment.log_metrics(metrics, step=step)

        self._log_drawbench(step, budget)

    def _log_drawbench(self, step: int, budget: int) -> None:
        """Evaluate DrawBench when enabled; must be called on all ranks."""
        # Image-only benchmark; each rank evaluates its own shard.
        if self.db_dataloader is not None:
            generated, prompts = self.generate_samples(
                self.db_dataloader,
                budget=budget,
                description='drawbench',
            )[2:4]

            with torch.inference_mode():
                metric = metrics_registry['IR'](
                    device=self.model.device
                ).calculate_dataset(
                    generated=generated,
                    prompts=prompts,
                    batch_size=self.config.batch_size,
                    reduction='sum',
                )

            sz = torch.tensor(len(prompts), device=metric.device)
            metric = ddp_reduce_sum(metric) / ddp_reduce_sum(sz)

            if self.is_main_process:
                self.experiment.log_metrics(
                    {'metrics/IR drawbench': metric}, step=step
                )

    def finalize(self) -> None:
        """End the Comet experiment."""
        if self.is_main_process:
            self.experiment.end()

    def log_learned_multistep(self) -> None:
        """Evaluate the learned schedule for several budgets.

        Logs to a separate Comet experiment in `log_multistep.project_name`,
        one test evaluation per budget in `log_multistep.budgets`. Must be
        called on all ranks.
        """
        if self.is_main_process:
            self.experiment = comet_ml.start(
                project_name=self.config.log_multistep.project_name
            )
            self.experiment.set_name(self.run_name)
            self.experiment.log_parameters(
                self.run_config, nested_support=True
            )

        for budget in tqdm(
            self.config.log_multistep.budgets,
            'multistep',
            disable=not self.is_main_process,
        ):
            self.log_test(budget, multistep=True)

        if self.is_main_process:
            self.experiment.end()
