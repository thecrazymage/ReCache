"""Evaluate sampled VBench videos with VBench, LPIPS and HPS.

Runs the standard VBench evaluation on all 16 dimensions and computes
the total score as VBench does. LPIPS to the teacher videos and HPS are
averaged over all frames.
"""

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

import packaging.version

if TYPE_CHECKING:
    from src.metrics.base import BaseMetric


EXPECTED_PROMPTS = 944
VIDEOS_PER_PROMPT = 5
FRAME_BATCH_SIZE = 8


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate generated videos with VBench."
    )
    parser.add_argument("device", help="Example: cuda:0")
    parser.add_argument("run_name", help="VBench generation run directory")
    parser.add_argument(
        "schedules", nargs="*", help="Schedule folders to evaluate"
    )
    parser.add_argument("--teacher-run", required=True)
    parser.add_argument("--all-schedules", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if not args.all_schedules and not args.schedules:
        parser.error("provide at least one schedule or pass --all-schedules")
    return args


def patch_vbench_imports() -> None:
    """Provide `pkg_resources.packaging`, which VBench's ViCLIP imports."""
    module = types.ModuleType("pkg_resources")
    module.packaging = packaging
    sys.modules.setdefault("pkg_resources", module)


def configure_cache(repo_root: Path) -> None:
    """Keep VBench, torch and HF downloads in `pretrained/vbench`."""
    cache_dir = (repo_root / "pretrained" / "vbench").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["VBENCH_CACHE_DIR"] = str(cache_dir)
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    os.environ["HF_HOME"] = str(cache_dir / "hf")
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hf_hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["DIFFUSERS_CACHE"] = str(cache_dir / "diffusers")


def load_prompts(full_info_path: Path | str) -> list[str]:
    """Load the unique prompts of the VBench prompt suite."""
    full_info = json.loads(Path(full_info_path).read_text(encoding="utf-8"))
    prompts = list(dict.fromkeys(item["prompt_en"] for item in full_info))
    if len(prompts) != EXPECTED_PROMPTS:
        raise RuntimeError(
            f"Expected {EXPECTED_PROMPTS} unique prompts, found {len(prompts)}"
        )
    return prompts


def validate_video_directory(videos_dir: Path, prompts: list[str]) -> None:
    """Check that the directory holds exactly the expected VBench videos."""
    expected = {
        f"{prompt}-{video_index}.mp4"
        for prompt in prompts
        for video_index in range(VIDEOS_PER_PROMPT)
    }
    actual = {
        path.name
        for path in videos_dir.glob("*.mp4")
        if path.is_file() and not path.name.endswith(".tmp.mp4")
    }
    if actual != expected:
        raise RuntimeError(
            f"Invalid video set in {videos_dir}: "
            f"{len(expected - actual)} missing, "
            f"{len(actual - expected)} unexpected"
        )


def calculate_frame_metrics(
    videos_dir: Path,
    teacher_dir: Path,
    prompts: list[str],
    lpips_metric: 'BaseMetric',
    hps_metric: 'BaseMetric',
) -> dict[str, float]:
    """Average LPIPS to the teacher videos and HPS over all frames."""
    from decord import VideoReader, cpu

    lpips_total = 0.0
    hps_total = 0.0
    frame_count = 0

    for prompt in tqdm(
        prompts, desc=f"{videos_dir.name} frame metrics", leave=False
    ):
        for video_index in range(VIDEOS_PER_PROMPT):
            filename = f"{prompt}-{video_index}.mp4"
            video_reader = VideoReader(
                str(videos_dir / filename), ctx=cpu(0), num_threads=1
            )
            teacher_reader = VideoReader(
                str(teacher_dir / filename), ctx=cpu(0), num_threads=1
            )

            if len(video_reader) != len(teacher_reader):
                raise RuntimeError(f"Frame count mismatch for {filename}")

            for start in range(0, len(video_reader), FRAME_BATCH_SIZE):
                frame_indices = range(
                    start,
                    min(start + FRAME_BATCH_SIZE, len(video_reader))
                )

                frames = (
                    torch.from_numpy(
                        video_reader.get_batch(frame_indices).asnumpy()
                    ).permute(0, 3, 1, 2).float().div_(255)
                )

                teacher_frames = (
                    torch.from_numpy(
                        teacher_reader.get_batch(frame_indices).asnumpy()
                    ).permute(0, 3, 1, 2).float().div_(255)
                )

                if frames.shape != teacher_frames.shape:
                    raise RuntimeError(f"Frame shape mismatch for {filename}")

                lpips_total += lpips_metric(
                    generated=frames,
                    references=teacher_frames,
                    reduction="sum"
                ).item()

                hps_total += hps_metric(
                    generated=frames,
                    prompts=[prompt] * len(frame_indices),
                    reduction="sum"
                ).item()

                frame_count += len(frame_indices)

    return {
        "LPIPS": lpips_total / frame_count,
        "HPS": hps_total / frame_count,
    }


def vbench_total_percent(scores: dict[str, list]) -> float:
    """Compute the VBench total score in percent.

    Same as `scripts/cal_final_score.py` in the VBench repository.
    """
    from VBench.scripts.constant import (
        DIM_WEIGHT,
        NORMALIZE_DIC,
        QUALITY_LIST,
        QUALITY_WEIGHT,
        SEMANTIC_LIST,
        SEMANTIC_WEIGHT,
    )

    raw = {
        key.replace("_", " "): float(value[0])
        for key, value in scores.items()
        if key != "total"
    }
    normalized = {}
    for key, value in raw.items():
        low = NORMALIZE_DIC[key]["Min"]
        high = NORMALIZE_DIC[key]["Max"]
        normalized[key] = (value - low) / (high - low) * DIM_WEIGHT[key]

    quality = sum(normalized[key] for key in QUALITY_LIST) / sum(
        DIM_WEIGHT[key] for key in QUALITY_LIST
    )
    semantic = sum(normalized[key] for key in SEMANTIC_LIST) / sum(
        DIM_WEIGHT[key] for key in SEMANTIC_LIST
    )
    total = (quality * QUALITY_WEIGHT + semantic * SEMANTIC_WEIGHT) / (
        QUALITY_WEIGHT + SEMANTIC_WEIGHT
    )
    return 100.0 * total


def main() -> None:
    """Evaluate the schedules of a VBench run."""
    args = parse_args()
    repo_root = Path.cwd()
    vbench_root = repo_root / "VBench"
    run_dir = vbench_root / "sampled_videos" / args.run_name
    teacher_dir = vbench_root / "sampled_videos" / args.teacher_run / "teacher"
    results_dir = (
        vbench_root / "sampled_videos/evaluation_results" / args.run_name
    )

    if args.all_schedules:
        schedule_dirs = sorted(
            (path
                for path in run_dir.iterdir()
                if path.is_dir() and path.name != "teacher"
            ), key=lambda path: path.name,
        )
    else:
        schedule_dirs = [run_dir / schedule for schedule in args.schedules]

    patch_vbench_imports()
    configure_cache(repo_root)
    sys.path.insert(0, str(vbench_root))

    from vbench import VBench

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index)

    full_info = vbench_root / "vbench" / "VBench_full_info.json"
    prompts = load_prompts(full_info)
    validate_video_directory(teacher_dir, prompts)

    from src.metrics.quality import HPSMetric
    from src.metrics.similarity import LPIPSMetric

    for videos_dir in schedule_dirs:
        validate_video_directory(videos_dir, prompts)

        schedule = videos_dir.name
        evaluator = VBench(device, str(full_info), str(results_dir))
        evaluator.evaluate(
            videos_path=str(videos_dir),
            name=schedule,
            dimension_list=evaluator.build_full_dimension_list(),
            mode="vbench_standard",
            local=args.local,
            read_frame=False,
        )

        result_path = results_dir / f"{schedule}_eval_results.json"
        (results_dir / f"{schedule}_full_info.json").unlink()
        scores = json.loads(result_path.read_text(encoding="utf-8"))
        total = vbench_total_percent(scores)
        print(f"{schedule}: total={total:.4f}")

        hps_metric = HPSMetric(device=device)
        lpips_metric = LPIPSMetric(device=device)

        frame_metrics = calculate_frame_metrics(
            videos_dir, teacher_dir, prompts,
            lpips_metric, hps_metric
        )
        scores = {"total": [total], **frame_metrics, **scores}
        result_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        result_path.replace(results_dir / f"{schedule}.json")

        print(
            f"{schedule}: LPIPS={frame_metrics['LPIPS']:.4f}, "
            f"HPS={frame_metrics['HPS']:.4f}"
        )


if __name__ == "__main__":
    main()
