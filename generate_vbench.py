"""Sample videos on the VBench prompt suite.

Follows the VBench sampling protocol: 5 videos per prompt, saved as
`<prompt>-<index>.mp4`, see
https://github.com/Vchitect/VBench/tree/master/prompts.
"""

import os
import json
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import click
import torch
import torch.distributed as dist
from diffusers.utils import export_to_video
from tqdm import tqdm

from src.caching_helpers import BaseCachingHelper, get_cacher_helper
from src.diffusion_wrappers import BaseDiffusionWrapper, get_diffusion_wrapper
from src.utils.config import config_from_path
from src.utils.ddp import ddp_end, ddp_setup
from src.utils.rng import seed_everything


OUTPUT_ROOT = Path("VBench/sampled_videos")
EXPECTED_PROMPTS = 944
VIDEOS_PER_PROMPT = 5


def load_prompts(
    path: Path | str = Path("VBench/vbench/VBench_full_info.json"),
) -> list[str]:
    """Load the unique prompts of the VBench prompt suite."""
    with Path(path).open("r", encoding="utf-8") as file:
        full_info = json.load(file)

    prompts = list(dict.fromkeys(item["prompt_en"] for item in full_info))
    assert (
        len(prompts) == EXPECTED_PROMPTS
    ), f"Expected {EXPECTED_PROMPTS} unique VBench prompts, got {len(prompts)}"
    return prompts


def target_from_schedule(
    schedule: str, nfe: int
) -> tuple[str, list[int], int]:
    """Turn comma-separated computed steps into a `(label, ts_to_skip, nfe)`
    target."""
    recompute = [int(step.strip()) for step in schedule.split(",")]

    if recompute[0] != 0:
        raise ValueError("A schedule must start with denoising step 0")
    if recompute != sorted(set(recompute)):
        raise ValueError(
            "Schedule steps must be unique and strictly increasing"
        )
    if any(step < 0 or step >= nfe for step in recompute):
        raise ValueError(f"Schedule steps must be between 0 and {nfe - 1}")

    recompute_set = set(recompute)
    skips = [step for step in range(nfe) if step not in recompute_set]
    label = ",".join(str(step) for step in recompute)
    return label, skips, nfe


def build_full_targets(
    teacher: bool, full_steps: Sequence[int], teacher_steps: int
) -> list[tuple[str, list[int], int]]:
    """Build targets that run full inference without caching."""
    if len(set(full_steps)) != len(full_steps):
        raise ValueError("Full-inference step counts must be unique")
    if any(steps < 1 for steps in full_steps):
        raise ValueError("Full-inference step counts must be positive")

    targets = [("teacher", [], teacher_steps)] if teacher else []
    targets.extend((f"full_{steps}", [], steps) for steps in full_steps)
    return targets


def generate_videos(
    model: BaseDiffusionWrapper,
    caching_helper: BaseCachingHelper | None,
    targets: list[tuple[str, list[int], int]],
    prompts: list[str],
    run_name: str,
    output_root: Path | str = OUTPUT_ROOT,
) -> None:
    """Sample videos for each target, skipping existing ones.

    Prompts are sharded across ranks, and the seed of each video depends
    only on its prompt and index.

    Args:
        targets: `(label, ts_to_skip, nfe)` tuples; videos go to
            `<output_root>/<run_name>/<label>/`.
    """
    from src.utils.logging.utils import to_video_frames

    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    seed_everything(42)
    model.eval()
    run_directory = Path(output_root) / run_name

    original_nfe = model.config.nfe
    try:
        for label, skips, nfe in targets:
            output_directory = run_directory / label
            output_directory.mkdir(parents=True, exist_ok=True)
            model.config.nfe = nfe

            for index in tqdm(
                range(rank, len(prompts), world_size),
                desc=f"VBench {label}",
                leave=False,
                disable=rank != 0,
            ):
                prompt = prompts[index]

                for video_index in range(VIDEOS_PER_PROMPT):
                    output_path = (
                        output_directory / f"{prompt}-{video_index}.mp4"
                    )
                    if output_path.exists():
                        continue

                    seed = index + video_index * len(prompts)
                    latent = torch.randn(
                        (1, *model.latents_shape),
                        dtype=model.dtype,
                        generator=torch.Generator("cpu").manual_seed(seed),
                    ).to(model.device)

                    inference_context = (
                        caching_helper.inference(ts_to_skip=skips)
                        if caching_helper is not None
                        else nullcontext()
                    )
                    with torch.inference_mode(), inference_context:
                        generated = model(
                            prompts=[prompt],
                            latents=latent,
                            output_type="pt",
                            output_video=True,
                        )

                    frames = to_video_frames(generated[0])
                    temporary_path = output_path.with_suffix(".tmp.mp4")
                    temporary_path.unlink(missing_ok=True)
                    export_to_video(
                        frames,
                        str(temporary_path),
                        fps=model.config.fps,
                        quality=5,
                    )
                    temporary_path.replace(output_path)

            if distributed:
                dist.barrier()
    finally:
        model.config.nfe = original_nfe


@click.command()
@click.option("--config", required=True, help="Path to YAML config file")
@click.option(
    "--run_name",
    "--run-name",
    "run_name",
    required=True,
    help="Output directory name in VBench/sampled_videos",
)
@click.option(
    "--device",
    default="cuda:0",
    help="Device for single-GPU runs, ignored under torchrun",
)
@click.option(
    "--schedule",
    multiple=True,
    help="Comma-separated computed steps, e.g. 0,4,8; repeatable",
)
@click.option("--teacher", is_flag=True, help="Sample with the teacher")
@click.option(
    "--full-step",
    "full_steps",
    type=int,
    multiple=True,
    help="Sample without caching with this many steps; repeatable",
)
def main(
    config: str,
    run_name: str,
    device: str,
    schedule: tuple[str, ...],
    teacher: bool,
    full_steps: tuple[int, ...],
) -> None:
    """Sample VBench videos for the teacher, full runs and schedules."""
    if not schedule and not teacher and not full_steps:
        raise click.UsageError(
            "at least one --schedule, --teacher, or --full-step is required"
        )

    config = config_from_path(config)

    nfe = config.diffusion_wrapper.parameters.nfe
    full_targets = build_full_targets(teacher, full_steps, nfe)
    schedule_targets = [target_from_schedule(value, nfe) for value in schedule]

    if schedule_targets and not hasattr(config, "caching_helper"):
        raise click.UsageError(
            "--schedule requires caching_helper in the config"
        )

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        local_rank, _, _ = ddp_setup()
        device = f"cuda:{local_rank}"

    prompts = load_prompts()
    model = get_diffusion_wrapper(
        config.diffusion_wrapper.wrapper_class,
        config.diffusion_wrapper.parameters,
        device
    )
    if full_targets:
        generate_videos(model, None, full_targets, prompts, run_name)

    if schedule_targets:
        caching_helper = get_cacher_helper(
            config.caching_helper.cacher_class,
            config.caching_helper.get('parameters', {}),
            model
        )
        generate_videos(
            model, caching_helper, schedule_targets, prompts, run_name
        )

    if distributed:
        ddp_end()


if __name__ == "__main__":
    main()
