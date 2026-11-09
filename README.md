# *ReCache*: Learning Budget-Aware Caching Schedules for Diffusion Models via REINFORCE

<p align="left">
<a href="https://arxiv.org/abs/2606.06060"><img src="https://img.shields.io/badge/arXiv-2606.06060-b31b1b.svg" alt="arXiv"></a>
<a href="https://github.com/thecrazymage/ReCache"><img src="https://img.shields.io/badge/Project-Page-green.svg" alt="Project Page"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-purple.svg" alt="License: Apache 2.0"></a>
</p>

<p align="center">
<a href="https://github.com/thecrazymage">Mishan&nbsp;Aliev</a><sup>∗,1</sup>, <a href="https://github.com/cocosinca">Eva&nbsp;Neudachina</a><sup>∗,1</sup>, <a href="https://github.com/thecrazymage/ReCache">Ilya&nbsp;Bykov</a><sup>∗,1</sup>, <a href="https://github.com/thecrazymage/ReCache">Aleksandr&nbsp;Oganov</a><sup>1</sup>,<br>
<a href="https://github.com/thecrazymage/ReCache">Kirill&nbsp;Struminsky</a><sup>2,1</sup>, <a href="https://github.com/thecrazymage/ReCache">Aibek&nbsp;Alanov</a><sup>1</sup>, <a href="https://github.com/thecrazymage/ReCache">Denis&nbsp;Rakitin</a><sup>1</sup>
</p>

<p align="center">
<sup>1</sup> HSE University, Russia<br>
<sup>2</sup> Yandex Research, Russia
</p>

<p align="center">
<sup>∗</sup> Equal contribution
</p>

![Visual comparison of ReCache and baseline generations](assets/comparison_grid.png)

### Abstract

Modern diffusion models generate high-quality images and videos, but their iterative denoising process makes inference expensive. Feature caching accelerates sampling by reusing or predicting intermediate activations across diffusion steps, exploiting computational redundancy along the reverse trajectory.

In this work, we align caching schedule selection with the goals of inference: preserving fidelity to uncached outputs and improving generation quality under a fixed budget of full model evaluations. We propose ***ReCache***, a policy-gradient method that learns a budget-conditioned scheduling policy to select which diffusion steps to fully recompute.

Training combines an output-fidelity objective with a generation-quality reward, requiring no labeled data or backpropagation through the full denoising trajectory.
ReCache supports both feature reuse and feature forecasting; for each model and caching mechanism, a single trained policy adapts across computational budgets at inference time without retraining.

## Table of Contents

- [Abstract](#abstract)
- [Installation](#installation)
  - [Environment](#environment)
  - [Metric Weights](#metric-weights)
- [Quickstart](#quickstart)
- [Reference Datasets](#reference-datasets)
  - [Configs](#configs)
  - [Output Layout](#output-layout)
- [Training](#training)
  - [Supported Caching Methods](#supported-caching-methods)
- [Evaluation](#evaluation)
  - [Baseline Evaluation](#baseline-evaluation)
  - [Video Evaluation with VBench](#video-evaluation-with-vbench)
- [Learned Policies](#learned-policies)
  - [FLUX.1-dev (N=50)](#flux1-dev-n50)
  - [Wan2.1 (N=25)](#wan21-n25)
  - [HunyuanVideo (N=50)](#hunyuanvideo-n50)
- [Reproducibility](#reproducibility)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Installation

Run all commands from the repository root: configs and code use paths relative to it (`data/`, `pretrained/`, `checkpoints/`).

### Environment

`configs/env.yaml` lists the exact dependencies; PyTorch is built for CUDA 12.8. Create and activate the environment with conda:

```bash
conda env create -f configs/env.yaml
conda activate recache
```

Diffusion models are downloaded from the Hugging Face Hub to `pretrained/models/` on first use. FLUX.1-dev is gated: accept its license on the model page and set your Hugging Face token. Training and baseline evaluation log to [Comet](https://www.comet.com/), so also set your Comet API key:

```bash
conda env config vars set HF_TOKEN=<your_hf_token> COMET_API_KEY=<your_comet_api_key>
conda activate recache  # reactivate the environment to apply the variables
```

VBench evaluation of video models needs a separate environment, see [Evaluation](#evaluation).

### Metric Weights

AQ, IQ and ImageReward load their weights from local files. Download them with:

```bash
bash scripts/download.sh
```

The other metrics download their weights automatically.

## Quickstart

To generate images using a caching method of your choice, find the corresponding config and use `get_models` to load the model and wrap it with the appropriate caching helper, which handles all caching operations.

The helper's `inference` context manager requires a `ts_to_skip` argument, specifying which timesteps to skip (timestep indices are 0-based).

For example, to generate an image with TaylorSeer caching on FLUX.1-dev, run:

```python
from torchvision.transforms.functional import to_pil_image

from src.utils.models import get_models
from src.utils.config import config_from_path


config = 'configs/flux/taylorseer_o1.yaml'
args = config_from_path(config)

model, caching_helper = get_models(
    args.diffusion_wrapper, args.caching_helper
)

prompt = ['A funny animal photo with absurd internet meme energy']

with caching_helper.inference(ts_to_skip=[5, 10, 15, 20, 25, 30]):
    image = model.forward(prompt)[0]
    
to_pil_image(image).save('output.png')

```


## Reference Datasets

Training and evaluation compare cached generations with generations of the same model at full NFE without caching. `generate_dataset.py` produces these references once per model: for every prompt it samples the initial noise, runs the full-NFE model and saves the prompt, the noise and the generation.

Pass the dataset config of your model (see [Configs](#configs)), for example for FLUX.1-dev:

```bash
python generate_dataset.py --config configs/flux/dataset.yaml
```

Options:

- `--device`: GPU to run the generation on, `cuda:0` by default. The script runs on a single GPU.
- `--seed`: base seed of the initial noises, `42` by default. Noises are sampled on CPU, so the dataset does not depend on the GPU.

### Configs

Each model has a dataset config in its directory under `configs/`:

| Model <br> Config | Resolution | Prompts | train / validation / test / DrawBench | Memory |
|---|---|---|---|---|
| FLUX.1-dev (NFE = 50) <br> [`configs/flux/dataset.yaml`](configs/flux/dataset.yaml) | 1024×1024 | [`data/coco.txt`](data/coco.txt) <br> (COCO captions) | 64 / 64 / 1000 / 200 | ~17 GB |
| Wan2.1 1.3B (NFE = 25) <br> [`configs/wan/dataset.yaml`](configs/wan/dataset.yaml) | 832×480, 81 frames | [`data/vbench_sampled.txt`](data/vbench_sampled.txt) <br> (VBench) | 64 / 64 / 128 / – | ~50 GB |
| HunyuanVideo (NFE = 50) <br> [`configs/hunyuan/dataset.yaml`](configs/hunyuan/dataset.yaml) | 640×480, 65 frames | [`data/vbench_sampled.txt`](data/vbench_sampled.txt) <br> (VBench) | 64 / 64 / 128 / – | ~31 GB |

To change the split sizes or the prompts, edit `train_size`, `val_size`, `test_size`, `prompts_path` or `db_prompts_path` in the config. A split of size `0` is skipped.

References are stored uncompressed in float32, and each split is kept in RAM until it is saved. The Wan test split needs about 100 GB of RAM.

### Output Layout

```
data/datasets/<model>/
├── config.yml           # copy of the dataset config
├── train/
│   ├── prompts.pkl      # list of N prompts
│   ├── noises.pt        # initial noises, (N, *latents_shape)
│   └── references.pt    # full-NFE generations in [0, 1]: (N, C, H, W), or (N, T, C, H, W) for video test splits
├── validation/
├── test/
└── drawbench/           # FLUX only
```

Training configs read the dataset from `training.dataset.path`, for example `data/datasets/flux` in `configs/flux/taylordp_o2.yaml`. The split sizes there must not exceed the generated ones; a smaller size takes the first samples of the split.

## Training

ReCache trains a budget-conditioned timestep policy. It is trained for one model and one caching method, and each training config fixes both. Training uses the reference dataset of the model, see [Reference Datasets](#reference-datasets).

For example, to train the scheduler for FLUX.1-dev with Δ-DiT caching, run:

```bash
python main.py --config configs/flux/deltaDiT.yaml
```

To train on several GPUs, launch `main.py` with `torch.distributed`:

```bash
python -m torch.distributed.run --nproc-per-node=8 main.py --config configs/wan/hicache_o1.yaml
```

Options:

- `--run_name`: run name in Comet.
- `--device`: GPU for single-GPU runs, `cuda:0` by default. Ignored under `torch.distributed`, use `CUDA_VISIBLE_DEVICES` variable instead.
- `--seed`: random seed, `42` by default. Overrides `training.seed` from the config.

### Supported Caching Methods

Training configs are in `configs/<model>/`. The `_o1` and `_o2` suffixes set the forecast order (`order` in `caching_helper.parameters`).

| Method | `cacher_class` | FLUX.1-dev (`configs/flux/`) | Wan2.1 1.3B (`configs/wan/`) | HunyuanVideo (`configs/hunyuan/`) |
|---|---|---|---|---|
| FORA | `FORACachingHelper` | [`fora.yaml`](configs/flux/fora.yaml) | – | – |
| Δ-DiT | `deltaDITCachingHelper` | [`deltaDiT.yaml`](configs/flux/deltaDiT.yaml) | – | – |
| DiCache | `DiCacheHelper` | [`dicache.yaml`](configs/flux/dicache.yaml) | – | – |
| LeMiCa | `LeMiCaHelper` | [`lemica.yaml`](configs/flux/lemica.yaml) | – | – |
| TaylorSeer | `TaylorSeerHelper` | [`taylorseer_o1.yaml`](configs/flux/taylorseer_o1.yaml), [`taylorseer_o2.yaml`](configs/flux/taylorseer_o2.yaml) | [`taylorseer_o1.yaml`](configs/wan/taylorseer_o1.yaml) | [`taylorseer_o1.yaml`](configs/hunyuan/taylorseer_o1.yaml) |
| HiCache | `HiCacheHelper` | [`hicache_o1.yaml`](configs/flux/hicache_o1.yaml), [`hicache_o2.yaml`](configs/flux/hicache_o2.yaml) | [`hicache_o1.yaml`](configs/wan/hicache_o1.yaml) | [`hicache_o1.yaml`](configs/hunyuan/hicache_o1.yaml) |
| Taylor-DP | `TaylorDPHelper` | [`taylordp_o2.yaml`](configs/flux/taylordp_o2.yaml) | [`taylordp_o1.yaml`](configs/wan/taylordp_o1.yaml) | [`taylordp_o1.yaml`](configs/hunyuan/taylordp_o1.yaml) |

## Evaluation

Training logs metrics on the training and validation splits. Every `test_frequency` epochs and after the last epoch, it also evaluates the test split and saves a checkpoint. Validation and these test evaluations use the budget set by `logging.parameters.eval_budget`.

After training, all processes evaluate the policy on their shards of the test set, and the averaged metrics are logged to Comet. The logging configuration is specified in `logging.parameters.log_multistep`. A separate Comet run is created in project with the name `project_name`, and metrics are evaluated using the budgets specified in `budgets`.

### Baseline Evaluation

To obtain metrics for full-NFE generation or supported caching methods evaluated with a uniform timestep schedule, use the `log_baselines.py` Python script. For example, to evaluate the LeMiCa caching method, run:

```bash
python log_baselines.py --config configs/flux/baselines/lemica.yaml
```

Options:

- `--run_name`: run name in Comet. Defaults to `recache training`.
- `--reference`: flag to enable full-NFE metric evaluation. If not set, the caching method specified in the config is used.
- `--device`: device to run the evaluation on. Defaults to `cuda:0`.
- `--seed`: random seed. Defaults to `42`.

### Video Evaluation with VBench

Evaluating VBench on video models requires a separate environment. Create and activate it as follows:

```bash
conda env create -f configs/vbench.yaml
conda activate vbench
bash scripts/setup_vbench.sh
```

The `scripts/setup_vbench.sh` script clones the VBench repository, installs it with detectron2 and downloads most of the models used by its metrics, at the revisions used to report the metrics in the paper. The remaining weights, such as CLIP, DINO, RAFT and MUSIQ, are downloaded during the first evaluation. Run the script on a machine with a GPU.

**Video generation.** To obtain VBench scores for your policy, first generate videos using the required prompts. Run the `generate_vbench.py` script in the `recache` environment. It generates 5 videos for each of the 944 prompts and saves them in `VBench/sampled_videos/<run-name>/<dir-name>/` directory.

The output directory depends on the flags passed to the `generate_vbench.py` script:

- `--teacher`: generates full-NFE videos and saves them in the teacher directory.
- `--full-step 7`: saves videos in the `full_7` directory. Generation uses no caching and only 7 steps.
- `--schedule "0,1,2,3,5,11,18"`: creates a directory named after the specified schedule, preserving the commas, and saves videos generated with those fully computed steps. The caching method is determined by the config.

For example, to generate Wan2.1 1.3B videos using the Taylor-DP caching method with `order=1` and a budget of 7, first generate the teacher videos, then generate the videos using the caching method.

1. **Generate teacher videos.** This step is required because VBench evaluation includes fidelity metrics that compare the generated videos against the teacher outputs.

   ```bash
   conda activate recache

   python -m torch.distributed.run --nproc-per-node=8 generate_vbench.py \
       --config configs/wan/teacher.yaml --run-name wan_teacher --teacher
   ```

2. **Generate videos using the caching method.** Once the teacher videos have been generated, run:

   ```bash
   python -m torch.distributed.run --nproc-per-node=8 generate_vbench.py \
       --config configs/wan/taylordp_o1.yaml --run-name wan_taylordp_o1 \
       --schedule "0,1,2,3,5,11,18"
   ```

The directory name depends only on the schedule, and existing videos are skipped, so use a separate `--run-name` for each caching method.

**Score evaluation.** VBench evaluation requires the `vbench` environment, so activate it and evaluate the scores as follows:

```bash
conda activate vbench

python evaluate_vbench.py cuda:0 \
    wan_teacher teacher \
    --teacher-run wan_teacher

python evaluate_vbench.py cuda:0 \
    wan_taylordp_o1 "0,1,2,3,5,11,18" \
    --teacher-run wan_teacher
```

Specify the desired device instead of `cuda:0` if needed.

The first command calculates metrics for the teacher run, so LPIPS will be 0. The second command calculates metrics for your policy. Both require `--teacher-run`, the run with the teacher videos that fidelity metrics compare against.

Scores are saved to `VBench/sampled_videos/evaluation_results/<run-name>/<schedule>.json`.

## Learned Policies

There are timestep policies learned with options and hyperparameters reported in the paper. You can use them for your evaluations without model training.

> **Note:** All timestep indices are 0-based.

### FLUX.1-dev (N=50)

| Caching method | k=7 | k=8 | k=9 | k=13 |
|---|---|---|---|---|
| FORA | `0, 1, 3, 6, 12, 24, 38` | `0, 1, 3, 6, 12, 24, 38, 43` | `0, 1, 3, 6, 12, 24, 38, 43, 47` | `0, 1, 2, 3, 4, 6, 8, 12, 24, 30, 38, 43, 47` |
| $\Delta$-DiT | `0, 1, 3, 10, 20, 34, 45` | `0, 1, 3, 5, 10, 20, 34, 45` | `0, 1, 3, 5, 10, 20, 34, 38, 45` | `0, 1, 2, 3, 5, 6, 8, 10, 20, 34, 38, 45, 48` |
| DiCache | `0, 1, 3, 6, 9, 18, 38` | `0, 1, 3, 6, 9, 18, 26, 38` | `0, 1, 3, 6, 9, 18, 26, 38, 48` | `0, 1, 2, 3, 4, 5, 6, 9, 12, 18, 26, 38, 48` |
| LeMiCa | `0, 1, 3, 10, 20, 33, 43` | `0, 1, 3, 6, 10, 20, 33, 43` | `0, 1, 3, 6, 10, 20, 33, 43, 47` | `0, 1, 2, 3, 5, 6, 8, 10, 15, 20, 33, 43, 47` |
| TaylorSeer (order=1) | `0, 3, 5, 10, 19, 30, 38` | `0, 1, 3, 5, 10, 19, 30, 38` | `0, 1, 3, 5, 10, 19, 30, 38, 48` | `0, 1, 2, 3, 4, 5, 7, 10, 14, 19, 30, 38, 48` |
| TaylorSeer (order=2) | `0, 3, 5, 10, 20, 35, 43` | `0, 3, 5, 10, 20, 27, 35, 43` | `0, 1, 3, 5, 10, 20, 27, 35, 43` | `0, 1, 2, 3, 5, 7, 10, 14, 20, 27, 35, 43, 48` |
| HiCache (order=1) | `0, 1, 3, 5, 10, 24, 38` | `0, 1, 3, 5, 10, 24, 38, 48` | `0, 1, 2, 3, 5, 10, 24, 38, 48` | `0, 1, 2, 3, 5, 7, 10, 14, 19, 24, 38, 43, 48` |
| HiCache (order=2) | `0, 1, 3, 10, 14, 24, 38` | `0, 1, 3, 5, 10, 14, 24, 38` | `0, 1, 3, 5, 10, 14, 24, 38, 48` | `0, 1, 2, 3, 5, 7, 10, 14, 19, 24, 38, 44, 48` |
| Taylor-DP (order=2) | `0, 3, 7, 14, 24, 35, 44` | `0, 1, 3, 7, 14, 24, 35, 44` | `0, 1, 3, 5, 7, 14, 24, 35, 44` | `0, 1, 2, 3, 4, 5, 7, 10, 14, 18, 24, 35, 44` |

### Wan2.1 (N=25)

| Caching method | k=7 | k=9 | k=13 |
|---|---|---|---|
| TaylorSeer (order=1) | `0, 1, 2, 4, 8, 13, 21` | `0, 1, 2, 3, 4, 6, 8, 13, 21` | `0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 18, 21, 24` |
| HiCache (order=1) | `0, 1, 2, 4, 8, 16, 21` | `0, 1, 2, 3, 4, 8, 16, 21, 24` | `0, 1, 2, 3, 4, 5, 6, 8, 11, 14, 16, 21, 24` |
| Taylor-DP (order=1) | `0, 1, 2, 3, 5, 11, 18` | `0, 1, 2, 3, 5, 8, 11, 18, 24` | `0, 1, 2, 3, 4, 5, 6, 8, 11, 18, 21, 23, 24` |

### HunyuanVideo (N=50)

| Caching method | k=7 | k=9 | k=13 |
|---|---|---|---|
| TaylorSeer (order=1) | `0, 1, 3, 8, 19, 29, 45` | `0, 1, 3, 5, 8, 12, 19, 29, 45` | `0, 1, 2, 3, 4, 5, 6, 8, 12, 19, 29, 45, 48` |
| HiCache (order=1) | `0, 1, 3, 6, 13, 19, 44` | `0, 1, 3, 6, 13, 19, 27, 44, 48` | `0, 1, 2, 3, 4, 5, 6, 8, 13, 19, 27, 44, 48` |
| Taylor-DP (order=1) | `0, 1, 3, 5, 11, 24, 39` | `0, 1, 3, 5, 8, 11, 24, 39, 47` | `0, 1, 2, 3, 4, 5, 6, 8, 11, 16, 24, 39, 47` |

## Reproducibility

Experimental details, including datasets, hyperparameters, and evaluation protocols, are described in the [paper](https://arxiv.org/abs/2606.06060) and its appendices. To reproduce the results, use the provided configurations and follow the setup instructions in this repository.

> **Note:** Computed metrics may differ slightly from those reported in the paper due to differences in hardware, software versions, and randomness in training and generation.

## Acknowledgements

We thank [Maxim Kodryan](https://www.youtube.com/watch?v=pKejn72xpCo), whose mere existence was contribution enough.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@misc{aliev2026recache,
  title         = {ReCache: Learning Budget-Aware Caching Schedules for Diffusion Models via REINFORCE},
  author        = {Aliev, Mishan and Neudachina, Eva and Bykov, Ilya and Oganov, Aleksandr and Struminsky, Kirill and Alanov, Aibek and Rakitin, Denis},
  year          = {2026},
  eprint        = {2606.06060},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2606.06060}
}
```
