#!/usr/bin/env python
from typing import Dict
import torch
import numpy as np
from pathlib import Path
import tqdm
import json
from typing import List
import math

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.regression_metrics import create_regression_metrics

from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.ensemble import symlink_checkpoint_files
from lib.files import prepare_results
from lib.serialization import serialize_human

# from lib.data_factory import register_dataset, get_factory
import lib.data_factory as data_factory
import lib.model_factory as model_factory
from dataclasses import dataclass
from lib.dataspec import DataSpec
from lib.data_utils import create_sample_legacy
from lib.train_dataclasses import TrainEpochState


class CliffordLinear(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.a = torch.nn.Parameter(
            torch.zeros(dim_out, dim_in, 8, dtype=torch.float32)
        )
        self.f = torch.nn.Parameter(
            torch.from_numpy(
                np.load(Path(__file__).parent / "./clifford_110_structure.npz"),
            ).float()
        )
        self.act_linear = torch.nn.Linear(8, 8, bias=False)
        self.f.requires_grad = False
        torch.nn.init.normal_(self.a, 0.0, math.sqrt(1.0 / (8 * dim_in)))

    def forward(self, x):
        # B, dim_in, 8
        # breakpoint()
        y = torch.einsum("mlj,jnr,bln->bmr", self.a, self.f, x)
        act = self.act_linear(y)
        act = torch.nn.functional.sigmoid(act)
        y = act * y
        # y = torch.nn.functional.relu(y)
        return y


@dataclass
class CliffordModelConfig:
    widths: List[int]

    def serialize_human(self):
        return serialize_human(self.__dict__)


class CliffordModel(torch.nn.Module):
    def __init__(self, config: CliffordModelConfig, data_spec: DataSpec):
        super().__init__()
        self.config = config
        assert data_spec.input_shape[-1] == 2
        assert data_spec.output_shape[-1] == 2
        self.cl_in = CliffordLinear(data_spec.input_shape[0], config.widths[0])
        in_out = list(zip(config.widths[0:], config.widths[1:]))
        self.cmlps = torch.nn.ModuleList(
            [CliffordLinear(in_dim, out_dim) for in_dim, out_dim in in_out]
        )
        self.cl_out = CliffordLinear(config.widths[-1], data_spec.output_shape[0])
        clifford_blades = json.loads(
            open(Path(__file__).parent / "clifford_110_blades.json").read()
        )
        self.basis = {tuple(key): idx for idx, key in enumerate(clifford_blades)}

    def forward(self, batch):
        x = batch["input"]
        cl_x = embed_clifford(x, self.basis)
        # breakpoint()
        y = self.cl_in(cl_x)
        # breakpoint()
        for idx, cl in enumerate(self.cmlps):
            y = cl(y)
            # breakpoint()
        y = self.cl_out(y)
        y = extract_clifford(y, self.basis)
        return dict(logits=y, predictions=y)


@dataclass
class DataCableConfig:
    npoints: int

    def serialize_human(self):
        return serialize_human(self.__dict__)


def length_cable(position):
    ps = position[1:] - position[:-1]
    return np.linalg.norm(ps, axis=-1).sum()


def step_function(position, distance_constraint, lr):
    # N, 3
    next_delta = position[1:] - position[:-1]
    next_delta = np.concatenate([next_delta, [[0, 0]]], axis=0)
    next_delta_len = np.linalg.norm(next_delta, axis=-1)
    prev_delta = position[1:] - position[:-1]
    prev_delta = np.concatenate([[[0, 0]], prev_delta], axis=0)
    prev_delta_len = np.linalg.norm(prev_delta, axis=-1)

    sufficiently_small_number = lr
    with np.errstate(divide="ignore", invalid="ignore"):
        dist_next_recip = (1.0 / next_delta_len)[:, None]
        dist_prev_recip = (1.0 / prev_delta_len)[:, None]
        # rand = np.random.rand(*dist_next_recip.shape) * 0.000001
        # dist_next_recip = np.where(np.isnan(dist_next_recip), rand, dist_next_recip)
        # dist_prev_recip = np.where(np.isnan(dist_prev_recip), rand, dist_prev_recip)
        next_grad = (
            np.concatenate([dist_next_recip] * 2, -1)
            * next_delta
            * -1.0
            * np.concatenate(
                [(next_delta_len - distance_constraint)[:, None]] * 2, axis=-1
            )
            * 2.0
        )
        prev_grad = (
            np.concatenate([dist_prev_recip] * 2, -1)
            * prev_delta
            * np.concatenate(
                [(prev_delta_len - distance_constraint)[:, None]] * 2, axis=-1
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


def embed_clifford(points, basis):
    # clifford_points = np.zeros(points.shape[0], 8)
    embedding = torch.zeros(8, 2, dtype=torch.float32, device=points.device)
    embedding[basis[(1,)], 0] = 1.0
    embedding[basis[(2,)], 1] = 1.0
    # breakpoint()
    clifford_points = torch.einsum(
        "ij,bnj->bni", embedding, points
    )  # embedding * points
    return clifford_points


def extract_clifford(cl_points, basis):
    debedding = torch.zeros(2, 8, dtype=torch.float32, device=cl_points.device)
    debedding[0, basis[(1,)]] = 1.0
    debedding[1, basis[(2,)]] = 1.0
    points = torch.einsum("ij,bnj->bni", debedding, cl_points)  # debedding * cl_points
    return points


class DataCable(torch.utils.data.Dataset):
    def __init__(self, data_config: DataCableConfig):
        self.config = data_config

    @staticmethod
    def data_spec(config: DataCableConfig):
        return DataSpec(
            input_shape=torch.Size([config.npoints - 1, 2]),
            output_shape=torch.Size([config.npoints, 2]),
            target_shape=torch.Size([config.npoints, 2]),
        )

    def __getitem__(self, idx):
        endpoints = np.random.rand(2, 2) * 5
        x = (
            np.linspace(endpoints[0, 0], endpoints[1, 0], 100, dtype=np.float32)[
                :, None
            ]
            # + np.cos(np.linspace(0, 2 * np.pi, 100))[:, None] * 2
        )
        y = (
            np.linspace(endpoints[0, 1], endpoints[1, 1], 100, dtype=np.float32)[
                :, None
            ]
            # + np.sin(np.linspace(0, 2 * np.pi, 100))[:, None] * 2
        )
        ps = np.concatenate([x, y], axis=-1)
        start_points = np.float32(np.random.rand(100, 2)) + ps
        # start_points = ps
        points = start_points
        for i in range(80):
            lr = 0.1  # + 0.3 * i / 8000  # min(0.4, 0.1 + 0.4 / 1000)
            points = step_function(points, 0.05, lr)
            # print(length_cable(points))
        start_deltas = start_points[1:] - start_points[:-1]
        start_to_end = points - start_points
        return create_sample_legacy(
            np.float32(start_deltas), np.float32(start_to_end), idx
        )

    def __len__(self):
        return 256 * 10


def create_config(ensemble_id):
    loss = torch.nn.L1Loss()

    def reg_loss(outputs, batch):
        return loss(outputs["logits"], batch["target"])

    train_config = TrainConfig(
        # model_config=MLPConfig(widths=[256, 256, 256]),
        model_config=CliffordModelConfig(widths=[256, 256]),
        train_data_config=DataCableConfig(npoints=100),
        val_data_config=DataCableConfig(npoints=100),
        loss=reg_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=3e-6, lr=5e-4),
        ),
        batch_size=256,
        ensemble_id=ensemble_id,
        _version=33,
    )
    train_eval = create_regression_metrics(torch.nn.functional.l1_loss, None)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=10),
        train_config=train_config,
        train_eval=train_eval,
        epochs=300,
        save_nth_epoch=1,
        validate_nth_epoch=5,
    )
    return train_run


if __name__ == "__main__":
    device_id = ddp_setup()

    # clifford_structure = torch.tensor(
    #     np.load(Path(__file__).parent / "./clifford_110_structure.npz"),
    #     dtype=torch.float32,
    # )
    # clifford_blades = json.loads(
    #     open(Path(__file__).parent / "clifford_110_blades.json").read()
    # )
    # basis = {tuple(key): idx for idx, key in enumerate(clifford_blades)}
    # v1 = torch.tensor([0.0] * 8)
    # v2 = torch.tensor([0.0] * 8)
    # v1[basis[(1,)]] = 1.0
    # v2[basis[(2,)]] = 1.0
    # w = torch.einsum("i,ikl,k->l", v1, clifford_structure, v2)
    data_factory.get_factory()
    data_factory.register_dataset(DataCableConfig, DataCable)

    mf = model_factory.get_factory()
    mf.register(CliffordModelConfig, CliffordModel)
    ensemble_config = create_ensemble_config(create_config, 1)
    ensemble = create_ensemble(ensemble_config, device_id)

    ds = data_factory.get_factory().create(DataCableConfig(npoints=100))
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=9,
        shuffle=False,
        drop_last=False,
    )

    result_path = prepare_results(
        Path(__file__).parent,
        f"{Path(__file__).stem}",
        ensemble_config,
    )
    symlink_checkpoint_files(ensemble, result_path)

    import matplotlib.pyplot as plt

    for xs, ys, ids in tqdm.tqdm(dl):
        xs = xs.to(device_id)

        output = ensemble.members[0](xs)["logits"].cpu().detach().numpy()
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        for idx, (start_deltas, delta, target) in enumerate(
            zip(xs.cpu().numpy(), output, ys.numpy())
        ):
            start = np.zeros_like(delta)
            # breakpoint()
            for i in range(start_deltas.shape[0]):
                start[i + 1] = start[i] + start_deltas[i]
            # delta = np.concatenate([[[0, 0]], delta], axis=0)
            ax = axs[idx // 3, idx % 3]
            ax.plot(start[:, 0], start[:, 1], "g--", label="initial", alpha=0.2)
            ax.plot(
                start[:, 0] + delta[:, 0],
                start[:, 1] + delta[:, 1],
                "r-",
                label="clifford",
            )
            ax.plot(
                start[:, 0] + target[:, 0],
                start[:, 1] + target[:, 1],
                "b-",
                label="target",
            )
            ax.legend()
            ax.set_title(f"{length_cable(start + target)}, {99 * 0.05}")
            fig.suptitle("80 iteration constraint resolve")
            fig.savefig("clifford_test.pdf")
        raise Exception("exit")
