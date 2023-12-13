import torch
from lib.train_dataclasses import ComputeConfig


def get_sampler(
    compute_config: ComputeConfig, ds: torch.utils.data.DataLoader, shuffle: bool
) -> (torch.utils.data.Sampler, bool):
    """Get device compatible sampler.

    Distributed data parallell dataloader need distributed sampler,
    """
    if compute_config.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(ds)
        shuffle = False
    else:
        sampler = None
    return sampler, shuffle


def create_sample_legacy(input, target, sample_id):
    return dict(input=input, target=target, sample_id=sample_id)
