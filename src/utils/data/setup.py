"""Dataloaders for the dataset splits."""

from munch import Munch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.utils.data.dataset import ReinforceDataset


class DistributedSamplerNoDuplicate(DistributedSampler):
    """Distributed sampler that does not pad the dataset.

    With `drop_last=False`, `DistributedSampler` pads the dataset by
    duplicating samples when its size is not divisible by the number of
    ranks, which biases evaluation metrics. This sampler instead lets
    some ranks get one sample fewer. Samples are not shuffled.

    Args:
        dataset: Dataset to sample from.
        num_replicas: Number of ranks.
        rank: Rank of the current process.
        seed: Passed to `DistributedSampler`; unused without shuffling.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=False,
        )
        if (not self.drop_last) and (
            len(self.dataset) % self.num_replicas != 0
        ):
            remainder = len(self.dataset) % self.num_replicas
            if self.rank >= remainder:
                self.num_samples -= 1
            self.total_size = len(self.dataset)


def _get_dataloader(
    path: str,
    size: int,
    batch_size: int,
    is_train: bool,
    distributed: bool,
    world_size: int,
    rank: int,
) -> DataLoader | None:
    """Build a dataloader for one split, or return None if `size` is 0."""
    if size <= 0:
        return None

    dataset = ReinforceDataset(path, dataset_size=size)

    sampler = None
    if distributed and is_train:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    elif distributed:
        sampler = DistributedSamplerNoDuplicate(
            dataset, num_replicas=world_size, rank=rank
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train and not distributed,
        sampler=sampler,
        collate_fn=ReinforceDataset.collate_fn,
        pin_memory=True,
    )


def get_dataloaders(
    config: Munch,
    batch_size: int,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> tuple[DataLoader | None, ...]:
    """Build dataloaders for the train, validation, test and DrawBench splits.

    A split of size 0 gets None instead of a dataloader. The train split
    is shuffled; under DDP, the other splits are sharded without
    duplicates, see `DistributedSamplerNoDuplicate`.

    Args:
        config: Dataset config, see `Config`.
        batch_size: Per-rank batch size.
        distributed: Whether to shard the data across ranks.
        world_size: Number of ranks.
        rank: Rank of the current process.

    Config:
        path (str): Dataset root with one subdirectory per split, as
            produced by `generate_dataset.py`.
        train_size (int): Number of train samples.
        val_size (int): Number of validation samples.
        test_size (int): Number of test samples.
        drawbench_size (int, optional): Number of DrawBench samples;
            missing means 0.

    Returns:
        Tuple `(train, validation, test, drawbench)` of dataloaders.
    """
    split_sizes = {
        'train': config.train_size,
        'validation': config.val_size,
        'test': config.test_size,
        'drawbench': config.get('drawbench_size', 0),
    }

    return tuple(
        _get_dataloader(
            f"{config.path}/{split}",
            size,
            batch_size,
            is_train=(split == 'train'),
            distributed=distributed,
            world_size=world_size,
            rank=rank,
        )
        for split, size in split_sizes.items()
    )
