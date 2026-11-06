"""Generate reference datasets with the full-NFE model, without caching."""

import os
import pickle

import yaml
import click
import torch
from tqdm import tqdm
from munch import Munch
from torch.utils.data import DataLoader

from src.utils.rng import seed_everything
from src.utils.config import config_from_path
from src.diffusion_wrappers import BaseDiffusionWrapper, get_diffusion_wrapper


def generate_and_save(
    config: Munch,
    model: BaseDiffusionWrapper,
    prompts: list[str],
    description: str,
    video: bool,
    seed: int,
) -> None:
    """Generate references for one split and save them to disk.

    Writes `prompts.pkl`, `noises.pt` and `references.pt` to
    `<save_path>/<description>/`, the layout read by `ReinforceDataset`.
    Noises are sampled on CPU, so the split does not depend on the device.

    Args:
        config: Dataset config; uses `save_path` and `batch_size`.
        model: Diffusion wrapper used to generate references.
        prompts: Prompts of the split.
        description: Split name, also used as the subdirectory name.
        video: Whether to save full videos for video models; if False,
            only the first frame is saved. Ignored by image models.
        seed: Seed for the initial noises.
    """
    generator = torch.Generator(device='cpu')

    noises = torch.randn(
        (len(prompts), *model.latents_shape),
        dtype=model.dtype,
        device='cpu',
        generator=generator.manual_seed(seed)
    )

    path = os.path.join(config.save_path, description)
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "prompts.pkl"), 'wb') as fp:
        pickle.dump(prompts, fp)

    torch.save(noises, os.path.join(path, "noises.pt"))

    dataloader = DataLoader(
        [*zip(noises, prompts)], batch_size=config.batch_size
    )

    references = []
    model.eval()
    for noise, prompt in tqdm(dataloader, f'{description} reference dataset'):
        noise = noise.to(model.device)
        prompt = list(prompt)

        with torch.inference_mode():
            references.append(
                model(
                    prompts=prompt, latents=noise, output_video=video
                ).to('cpu')
            )

    references = torch.concat(references, dim=0)
    torch.save(references, os.path.join(path, "references.pt"))


def generate_dataset(config: Munch, seed: int, device: str) -> None:
    """Generate all dataset splits described by the config.

    `train` and `validation` take consecutive prompts from the start of
    the file, `test` takes the last `test_size` ones. Splits use noise
    seeds `seed`, `seed + 1`, `seed + 2` and `seed + 3` (drawbench). For
    video models only `test` keeps full videos.

    Args:
        config: Dataset config, see `configs/<model>/dataset.yaml`.
        seed: Base seed for the initial noises.
        device: Device to run the diffusion model on.
    """
    wrapper_config = config.diffusion_wrapper
    model = get_diffusion_wrapper(
        wrapper_config.wrapper_class, wrapper_config.parameters, device
    )

    with open(config.prompts_path, 'r') as f:
        prompts = [line.strip() for line in f]

    if config.train_size > 0:
        train_prompts = prompts[:config.train_size]
        generate_and_save(
            config, model, train_prompts, 'train', video=False, seed=seed
        )

    if config.val_size > 0:
        start, end = config.train_size, config.train_size + config.val_size
        val_prompts = prompts[start:end]
        generate_and_save(
            config, model, val_prompts, 'validation', video=False, seed=seed+1
        )

    if config.test_size > 0:
        test_prompts = prompts[-config.test_size:]
        video = model.is_video
        generate_and_save(
            config, model, test_prompts, 'test', video=video, seed=seed+2
        )

    db_prompts_path = config.get('db_prompts_path', None)
    if db_prompts_path:
        with open(db_prompts_path, 'r') as f:
            db_prompts = f.read().split('\n')

        generate_and_save(
            config, model, db_prompts, 'drawbench', video=False, seed=seed+3
        )

    with open(os.path.join(config.save_path, 'config.yml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


@click.command()
@click.option(
    "--config", type=str, required=True, help="Path to dataset config file"
)
@click.option("--seed", type=int, default=42, help="Base random seed")
@click.option(
    "--device", type=str, default="cuda:0", help="Device for generation"
)
def main(config: str, seed: int, device: str) -> None:
    """Generate reference datasets from a dataset YAML config."""
    seed_everything(seed)
    generate_dataset(config_from_path(config), seed, device)


if __name__ == '__main__':
    main()
