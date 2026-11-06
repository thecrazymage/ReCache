"""Dataset of prompts, initial noises and full-NFE references."""

import os
import pickle

import torch
from torch.utils.data import Dataset


class ReinforceDataset(Dataset):
    """One dataset split produced by `generate_dataset.py`.

    Loads `prompts.pkl`, `noises.pt` and `references.pt` from the split
    directory. Each item is a `(prompt, noise, reference)` tuple.

    Args:
        path_to_dataset: Split directory, e.g. `data/datasets/flux/train`.
        dataset_size: Number of first samples to keep, at most the split
            size; None or 0 keeps all.
    """

    def __init__(
        self, path_to_dataset: str, dataset_size: int | None = None
    ) -> None:
        super().__init__()

        with open(os.path.join(path_to_dataset, "prompts.pkl"), 'rb') as f:
            self.prompts = pickle.load(f)

        if dataset_size:
            dataset_size = min(dataset_size, len(self.prompts))
        else:
            dataset_size = len(self.prompts)
        print(
            f'loading {path_to_dataset.split("/")[-1]} '
            f'dataset of size {dataset_size}'
        )

        self.prompts = self.prompts[:dataset_size]

        self.noises = torch.load(
            os.path.join(path_to_dataset, "noises.pt"), map_location='cpu'
        )[:dataset_size]

        self.references = torch.load(
            os.path.join(path_to_dataset, "references.pt"), map_location='cpu'
        )[:dataset_size]

        assert (
            len(self.prompts) == self.noises.shape[0]
        ), "prompts and noises don't match in length"
        assert (
            len(self.prompts) == self.references.shape[0]
        ), (
            "full inference reference images doesn't match in length with "
            "prompts and noises"
        )

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(
        self, index: int
    ) -> tuple[str, torch.Tensor, torch.Tensor]:
        return (
            self.prompts[index],
            self.noises[index],
            self.references[index],
        )

    @classmethod
    def collate_fn(
        cls, batch: list[tuple[str, torch.Tensor, torch.Tensor]]
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """Collate items into a `(prompts, noises, references)` batch."""
        prompts, noises, teacher = zip(*batch)
        return (
            list(prompts),
            torch.stack(noises, dim=0),
            torch.stack(teacher, dim=0),
        )
