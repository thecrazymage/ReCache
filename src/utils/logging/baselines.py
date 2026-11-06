"""Comet logger for caching baselines with fixed step schedules."""

import ast
import os

import comet_ml
import yaml
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from munch import Munch
from torch.utils.data import DataLoader

from src.metrics import metrics_registry
from src.caching_helpers import BaseCachingHelper
from src.diffusion_wrappers import BaseDiffusionWrapper
from src.utils.logging.utils import (
    prepare_images,
    draw_timesteps_plot,
    calc_metrics,
    reverse_timesteps,
)
from src.utils.logging.base import BaseLogger


class BaselinesLogger(BaseLogger):
    """Logger that evaluates fixed step schedules and the teacher.

    Used by `log_baselines.py`: evaluates a caching method with the
    uniform schedules from `uniform_path`, or the full-NFE teacher, on
    the test set and logs the results to Comet. Runs in a single process.

    Unlike `BaseLogger`, does not create a timestamped run directory:
    files go to `checkpoint_path` as is, if it is set.

    Args:
        config: Full baseline config; logger settings are read from
            `config.logging.parameters`.
        run_name: Comet experiment name.

    Config:
        project_name (str): Comet project.
        uniform_path (str): File with uniform schedules, a Python dict
            literal `{nfe: {budget: [computed steps]}}`.
        budgets (list[int]): Budgets to evaluate.
        quality_metrics (list[str]): Quality metric names in
            `metrics_registry`.
        fidelity_metrics (list[str]): Fidelity metric names in
            `metrics_registry`.
        batch_size (int): Batch size for metric computation.
        log_size (int): Number of samples in logged image grids and
            saved videos.
        checkpoint_path (str, optional): Directory for the config dump
            and videos.
    """

    def __init__(self, config: Munch, run_name: str) -> None:
        self.is_main_process = True

        self.run_name = run_name
        self.config = config.logging.parameters

        self.experiment = comet_ml.start(project_name=self.config.project_name)

        self.experiment.set_name(run_name)
        self.experiment.log_parameters(config, nested_support=True)

        checkpoint_path = self.config.get("checkpoint_path")
        if checkpoint_path:
            os.makedirs(checkpoint_path, exist_ok=True)

            config_path = os.path.join(checkpoint_path, "config.yaml")
            with open(config_path, "w") as f:
                yaml.safe_dump(
                    config.toDict(), f, sort_keys=False, allow_unicode=True
                )

        print(
            f"""
            🍄 Logger initialized!
            Run name: {self.run_name}
            Checkpoint path: {checkpoint_path}
        """
        )

    def setup(
        self,
        model: BaseDiffusionWrapper,
        caching_helper: BaseCachingHelper | None,
        dataloader: DataLoader,
        db_dataloader: DataLoader | None,
    ) -> None:
        """Attach the model and data and load the uniform schedules.

        Args:
            model: Diffusion wrapper.
            caching_helper: Caching helper, None for teacher runs.
            dataloader: Test dataloader.
            db_dataloader: DrawBench dataloader, or None.
        """
        self.model = model
        self.caching_helper = caching_helper
        self.dataloader = dataloader
        self.db_dataloader = db_dataloader

        with open(self.config.uniform_path, 'r') as file:
            steps = ast.literal_eval(file.read())

        self.uniform_timesteps = steps[self.model.config.nfe]

    @torch.no_grad()
    def log_test(self, budget: int, ts_to_skip: list[int]) -> None:
        """Evaluate a fixed schedule on the test set.

        Logs test metrics, a sample grid and the schedule plot. 
        For video models, also saves sample videos 
        if `checkpoint_path` is set.

        Args:
            budget: Evaluation budget, is used as the Comet step.
            ts_to_skip: Indices of steps to skip.
        """
        generated, prompts, references = self.generate_samples(
            self.dataloader,
            ts_to_skip=ts_to_skip,
            output_video=self.model.is_video,
        )[-3:]

        if self.model.is_video:
            if self.config.get("checkpoint_path"):
                self.save_video(
                    budget, generated[-self.config.log_size:]
                )
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
            image_data=grid, name='test/samples', step=budget
        )

        fig = draw_timesteps_plot(
            ts_to_skip, self.model.config.nfe, budget
        )
        self.experiment.log_figure(
            figure=fig, figure_name='test/timesteps_plot', step=budget
        )
        plt.close(fig)

        metric_names = self.config.fidelity_metrics + self.config.quality_metrics

        with torch.inference_mode():
            metrics = calc_metrics(
                metric_names=metric_names,
                device=self.model.device,
                generated=generated,
                references=references,
                prompts=prompts,
                batch_size=self.config.batch_size,
                is_video=self.model.is_video,
            )

        print(
            f'\n\n🏔  test metrics for budget {budget} calculated!\n'
        )
        for name, value in metrics.items():
            print(name.split('/')[-1], value)
        print('\n')

        self.experiment.log_metrics(metrics, step=budget)

        # ------------------------- drawbench -------------------------
        # calculated only for t2i, thus single-threaded
        if self.db_dataloader is not None:
            generated, prompts = self.generate_samples(
                self.db_dataloader,
                ts_to_skip=ts_to_skip,
                description='drawbench',
            )[2:4]

            with torch.inference_mode():
                metric = metrics_registry['IR'](
                    device=self.model.device
                ).calculate_dataset(
                    generated=generated,
                    prompts=prompts,
                    batch_size=self.config.batch_size,
                )

            self.experiment.log_metrics(
                {'metrics/IR brawbench': metric}, step=budget
            )

        # ----------------------------------------------------------------

    def finalize(self) -> None:
        """End the Comet experiment."""
        self.experiment.end()

    def log_multistep(self) -> None:
        """Evaluate the uniform schedule for each budget in `budgets`."""
        for budget in tqdm(self.config.budgets, 'multistep'):
            ts_to_skip = reverse_timesteps(
                self.uniform_timesteps[budget], nfes=self.model.config.nfe
            )[1]

            self.log_test(budget, ts_to_skip)

    @torch.no_grad()
    def log_teacher(self) -> None:
        """Evaluate the full-NFE teacher outputs stored in the dataset.

        The same values are logged at every budget in `budgets`, so they plot 
        as a horizontal line next to the baselines.
        """
        prompts = self.dataloader.dataset.prompts
        references = self.dataloader.dataset.references

        if self.model.is_video and self.config.get("checkpoint_path"):
            self.save_video('teacher', references[-self.config.log_size:])

        with torch.inference_mode():
            metrics = calc_metrics(
                metric_names=self.config.quality_metrics,
                device=self.model.device,
                generated=references,
                prompts=prompts,
                batch_size=self.config.batch_size,
                is_video=self.model.is_video,
            )

        print(f'\n\n🏔  test metrics for teacher calculated!\n')
        for name, value in metrics.items():
            print(name.split('/')[-1], value)
        print('\n')

        for budget in self.config.budgets:
            self.experiment.log_metrics(metrics, step=budget)

        # ------------------------- drawbench -------------------------
        # calculated only for t2i, thus single-threaded
        if self.db_dataloader is not None:
            prompts = self.db_dataloader.dataset.prompts
            references = self.db_dataloader.dataset.references

            with torch.inference_mode():
                metric = metrics_registry['IR'](
                    device=self.model.device
                ).calculate_dataset(
                    generated=references,
                    prompts=prompts,
                    batch_size=self.config.batch_size,
                )

            print(f'\nIR drawbench {metric}\n')

            for budget in self.config.budgets:
                self.experiment.log_metrics(
                    {'metrics/IR brawbench': metric}, step=budget
                )

        # ----------------------------------------------------------------
