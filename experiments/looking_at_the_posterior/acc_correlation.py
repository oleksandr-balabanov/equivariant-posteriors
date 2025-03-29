#!/usr/bin/env python
import ssl

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import binned_statistic_2d

from lib.data_registry import DataCIFARConfig
from lib.data_registry import DataCIFAR10CConfig
from lib.data_registry import DataCIFAR10C
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
from experiments.looking_at_the_posterior.uq import uq_for_ensemble, uncertainty

ssl._create_default_https_context = ssl._create_unverified_context


def accs_and_inferred_accs(dl_uq_calibration, dl_pool, ensemble, bins: int, device_id):
    uq_calibration = uncertainty(dl_uq_calibration, ensemble, device_id)
    acc = (
        torch.where(
            uq_calibration.targets[:, None].cpu()
            == uq_calibration.mean_pred[:, None].cpu(),
            1.0,
            0.0,
        )
        .numpy()
        .squeeze()
    )
    # log_H = torch.log(uq_calibration.H)
    # log_MI = torch.log(uq_calibration.MI)
    mean_acc_calibration, x_bins, y_bins, bin_number = binned_statistic_2d(
        uq_calibration.H.cpu().numpy(),
        uq_calibration.MI.cpu().numpy(),
        acc,
        statistic="mean",
        bins=bins,
        expand_binnumbers=True,
    )
    cal_counts, _, _, _ = binned_statistic_2d(
        uq_calibration.H.cpu().numpy(),
        uq_calibration.MI.cpu().numpy(),
        acc,
        statistic="count",
        bins=[x_bins, y_bins],
        expand_binnumbers=True,
    )

    uq_pool = uncertainty(dl_pool, ensemble, device_id)
    acc = (
        torch.where(
            uq_pool.targets[:, None].cpu() == uq_pool.mean_pred[:, None].cpu(),
            1.0,
            0.0,
        )
        .numpy()
        .squeeze()
    )
    # log_H = torch.log(uq_pool.H)
    # log_MI = torch.log(uq_pool.MI)
    mean_acc_pool, x_bins, y_bins, bin_number_pool = binned_statistic_2d(
        uq_pool.H.cpu().numpy(),
        uq_pool.MI.cpu().numpy(),
        acc,
        statistic="mean",
        bins=[x_bins, y_bins],
        expand_binnumbers=True,
    )

    bin_coords = list(zip(bin_number_pool[0] - 1, bin_number_pool[1] - 1))
    bin_coords_and_sample_ids = list(zip(bin_coords, uq_pool.sample_ids))
    bin_coords_and_sample_ids_with_values = [
        (coord, sample_id)
        for (coord, sample_id) in bin_coords_and_sample_ids
        if (
            coord[0] >= 0
            and coord[0] < len(x_bins) - 1
            and coord[1] >= 0
            and coord[1] < len(y_bins) - 1
        )
    ]

    just_coords_x = [coord[0] for coord, _ in bin_coords_and_sample_ids_with_values]
    just_coords_y = [coord[1] for coord, _ in bin_coords_and_sample_ids_with_values]
    # just_sample_ids = [
    # sample_id for _, sample_id in bin_coords_and_sample_ids_with_values
    # ]

    inferred_accs = mean_acc_calibration[just_coords_x, just_coords_y]

    return mean_acc_pool, inferred_accs


if __name__ == "__main__":
    device_id = ddp_setup()

    ensemble_config_conv = create_ensemble_config(
        create_config_function(model_config=ConvLAPConfig(), batch_size=3000),
        n_members=10,
    )
    ensemble_conv = create_ensemble(ensemble_config_conv, device_id)

    ds_cifar_val = data_factory.get_factory().create(DataCIFARConfig(validation=True))
    dl_cifar_val = torch.utils.data.DataLoader(
        ds_cifar_val,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )

    # ds_cifar_c = data_factory.get_factory().create(
    #     create_corrupted_dataset_config()
    #     # DataCIFAR10CConfig(subset="impulse_noise", severity=1)
    # )
    # dl_cifar_c = torch.utils.data.DataLoader(
    #     ds_cifar_c,
    #     batch_size=256,
    #     shuffle=False,
    #     drop_last=False,
    # )

    all_subsets = DataCIFAR10C.cifarc_subsets[:]
    even_subset_config = DataCIFAR10CConfig(
        subsets=all_subsets[::2], severities=[1, 2, 3, 4, 5]
    )
    ds_cifar_c_even = data_factory.get_factory().create(even_subset_config)
    dl_cifar_c_even = torch.utils.data.DataLoader(
        ds_cifar_c_even,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )
    odd_config = DataCIFAR10CConfig(
        subsets=all_subsets[1::2], severities=[1, 2, 3, 4, 5]
    )
    ds_cifar_c_odd = data_factory.get_factory().create(odd_config)
    dl_cifar_c_odd = torch.utils.data.DataLoader(
        ds_cifar_c_odd,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )

    mean_acc_pool, inferred_accs = accs_and_inferred_accs(
        dl_cifar_c_even, dl_cifar_c_odd, ensemble_conv, 15, device_id
    )

    df = pd.DataFrame(
        data=list(
            zip(mean_acc_pool.flatten().tolist(), inferred_accs.flatten().tolist())
        ),
        columns=["pool_acc", "inferred_acc"],
    )
    df.to_csv(Path(__file__).parent / "inferred_accs_cifar10c.csv")
    # breakpoint()

    # uq_for_ensemble(dl_cifar_c, ensemble_conv, ensemble_config_conv, "conv", device_id)
    # uq_for_ensemble(
    # dl_cifar_val, ensemble_conv, ensemble_config_conv, "conv", device_id
    # )
