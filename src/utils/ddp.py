"""DDP process group setup, teardown and cross-rank reductions."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def ddp_setup() -> tuple[int, int, int]:
    """Initialize the NCCL process group from torchrun env variables.

    Sets the current CUDA device to `LOCAL_RANK`.

    Returns:
        Tuple `(local_rank, rank, world_size)`.
    """

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(hours=2),
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return local_rank, rank, world_size


def ddp_end() -> None:
    """Wait for all ranks and destroy the process group."""
    dist.barrier()
    dist.destroy_process_group()


def ddp_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum a tensor over all ranks; return it unchanged without DDP."""
    if not (dist.is_available() and dist.is_initialized()):
        return x

    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


def ddp_reduce_mean(x: torch.Tensor, world_size: int) -> torch.Tensor:
    """Average a tensor over all ranks; return it unchanged without DDP."""
    if not (dist.is_available() and dist.is_initialized()):
        return x

    return ddp_reduce_sum(x) / world_size
