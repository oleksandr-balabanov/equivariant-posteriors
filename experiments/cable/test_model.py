#!/usr/bin/env python
import torch
import numpy as np
from pathlib import Path
import tqdm

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.regression_metrics import create_regression_metrics

from lib.models.mlp import MLPConfig
from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.ensemble import symlink_checkpoint_files
from lib.files import prepare_results
from lib.serialization import serialize_human

from lib.data_factory import register_dataset, get_factory
from dataclasses import dataclass
from lib.dataspec import DataSpec


@dataclass
class DataCableConfig:
    npoints: int

    def serialize_human(self):
        return serialize_human(self.__dict__)


def length_cable(position):
    ps = position[1:] - position[:-1]
    return np.linalg.norm(ps, axis=-1).sum()


def step_function(position, distance_constraint):
    # N, 3
    next_delta = position[1:] - position[:-1]
    next_delta = np.concatenate([next_delta, [[0, 0, 0]]], axis=0)
    next_delta_len = np.linalg.norm(next_delta, axis=-1)
    prev_delta = position[1:] - position[:-1]
    prev_delta = np.concatenate([[[0, 0, 0]], prev_delta], axis=0)
    prev_delta_len = np.linalg.norm(prev_delta, axis=-1)

    sufficiently_small_number = 0.1
    with np.errstate(divide="ignore", invalid="ignore"):
        dist_next_recip = (1.0 / next_delta_len)[:, None]
        dist_prev_recip = (1.0 / prev_delta_len)[:, None]
        # rand = np.random.rand(*dist_next_recip.shape) * 0.000001
        # dist_next_recip = np.where(np.isnan(dist_next_recip), rand, dist_next_recip)
        # dist_prev_recip = np.where(np.isnan(dist_prev_recip), rand, dist_prev_recip)
        next_grad = (
            np.concatenate([dist_next_recip] * 3, -1)
            * next_delta
            * -1.0
            * np.concatenate(
                [(next_delta_len - distance_constraint)[:, None]] * 3, axis=-1
            )
            * 2.0
        )
        prev_grad = (
            np.concatenate([dist_prev_recip] * 3, -1)
            * prev_delta
            * np.concatenate(
                [(prev_delta_len - distance_constraint)[:, None]] * 3, axis=-1
            )
            * 2.0
        )
        grad = next_grad + prev_grad
        translate = sufficiently_small_number * -grad
        rand = np.random.rand(*translate.shape) * 0.000001
        translate = np.where(np.isnan(translate), rand, translate)
        translate[0, :] = 0
        translate[-1, :] = 0
    return position + translate


class DataCable(torch.utils.data.Dataset):
    def __init__(self, data_config: DataCableConfig):
        self.config = data_config

    @staticmethod
    def data_spec(config: DataCableConfig):
        return DataSpec(
            input_shape=torch.Size([config.npoints - 1, 3]),
            output_shape=torch.Size([config.npoints, 3]),
            target_shape=torch.Size([config.npoints, 3]),
        )

    def __getitem__(self, idx):
        start_points = np.float32(np.random.rand(100, 3))
        points = start_points
        for i in range(80):
            points = step_function(points, 0.05)
            # print(length_cable(points))
        start_deltas = start_points[1:] - start_points[:-1]
        start_to_end = points - start_points
        return np.float32(start_deltas), np.float32(start_to_end), idx

    def __len__(self):
        return 256


def create_config(ensemble_id):
    loss = torch.nn.L1Loss()

    def reg_loss(outputs, targets):
        return loss(outputs["logits"], targets)

    train_config = TrainConfig(
        model_config=MLPConfig(widths=[256, 256, 256]),
        train_data_config=DataCableConfig(npoints=100),
        val_data_config=DataCableConfig(npoints=100),
        loss=reg_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=3e-6, lr=5e-4),
        ),
        batch_size=256,
        ensemble_id=ensemble_id,
        _version=29,
    )
    train_eval = create_regression_metrics(torch.nn.functional.l1_loss, None)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=10),
        train_config=train_config,
        train_eval=train_eval,
        epochs=5000,
        save_nth_epoch=1,
        validate_nth_epoch=5,
    )
    return train_run


if __name__ == "__main__":
    # x = np.linspace(0, 1.0, 10) + np.random.rand(10)
    # y = np.linspace(0, 1.0, 10)
    # z = np.zeros(10)
    # p = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=-1)
    # for _ in range(100):
    #     p = step_function(p, 0.1)
    #     print(p)
    #     input()
    device_id = ddp_setup()

    get_factory()
    register_dataset(DataCableConfig, DataCable)
    ensemble_config = create_ensemble_config(create_config, 1)
    ensemble = create_ensemble(ensemble_config, device_id)

    ds = get_factory().create(DataCableConfig(npoints=100))
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    result_path = prepare_results(
        Path(__file__).parent,
        f"{Path(__file__).stem}",
        ensemble_config,
    )
    symlink_checkpoint_files(ensemble, result_path)

    for xs, ys, ids in tqdm.tqdm(dl):
        xs = xs.to(device_id)

        output = ensemble.members[0](xs)
