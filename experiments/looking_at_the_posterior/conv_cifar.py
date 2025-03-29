#!/usr/bin/env python
import ssl

import torch

from lib.data_factory import DataCIFARConfig
import lib.data_factory as data_factory
import lib.slurm as slurm

from lib.models.conv_lap import ConvLAPConfig

from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.ensemble import train_member

from experiments.looking_at_the_posterior.config import (
    create_config_function,
    create_corrupted_dataset_config,
)
from experiments.looking_at_the_posterior.uq import uq_for_ensemble

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == "__main__":
    device_id = ddp_setup()

    ensemble_config_conv = create_ensemble_config(
        create_config_function(model_config=ConvLAPConfig(), batch_size=3000),
        n_members=20,
    )
    if slurm.get_task_id() is not None:
        train_member(ensemble_config_conv, slurm.get_task_id(), device_id)
    else:
        ensemble_conv = create_ensemble(ensemble_config_conv, device_id)

    if slurm.get_task_id() is not None:
        print("Exiting early since this is a SLURM array job used for training only")
        exit(0)

    ds_cifar_val = data_factory.get_factory().create(DataCIFARConfig(validation=True))
    dl_cifar_val = torch.utils.data.DataLoader(
        ds_cifar_val,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )

    ds_cifar_c = data_factory.get_factory().create(
        create_corrupted_dataset_config()
        # DataCIFAR10CConfig(subset="impulse_noise", severity=1)
    )
    dl_cifar_c = torch.utils.data.DataLoader(
        ds_cifar_c,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )

    uq_for_ensemble(dl_cifar_c, ensemble_conv, ensemble_config_conv, "conv", device_id)
    uq_for_ensemble(
        dl_cifar_val, ensemble_conv, ensemble_config_conv, "conv", device_id
    )
