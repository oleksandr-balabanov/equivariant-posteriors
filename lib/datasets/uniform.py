from typing import Dict
import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec
from lib.data_utils import create_sample_legacy
from lib.train_dataclasses import TrainEpochState


@dataclass(frozen=True)
class DataUniformConfig:
    min: float
    max: float
    N: int

    def serialize_human(self):
        return dict(min=self.min, max=self.max, N=self.N)


@dataclass
class Uniform:
    xs: torch.Tensor
    ys: torch.Tensor
    sample_ids: torch.Tensor


def generate_uniform_points(min, max, N):
    x, y = torch.meshgrid(
        torch.linspace(min, max, steps=N), torch.linspace(min, max, steps=N)
    )
    x = x.flatten()[:, None]
    y = y.flatten()[:, None]
    xs = torch.concat([x, y], dim=-1)
    return Uniform(
        xs=xs[:, :, None],
        ys=torch.zeros_like(x),
        sample_ids=torch.arange(0, x.shape[0], 1, dtype=torch.int32),
    )


class DataUniform(torch.utils.data.Dataset):
    def __init__(self, data_config: DataUniformConfig):
        self.uniform = generate_uniform_points(
            data_config.min, data_config.max, data_config.N
        )
        self.n_classes = 2
        self.config = data_config

    def data_spec(self):
        return DataSpec(
            input_shape=self.uniform.xs.shape[1:],
            target_shape=self.uniform.ys.shape[1:],
            output_shape=torch.Size([1]),
        )

    @staticmethod
    def sample_id_spec(config):
        return ["idx"]

    def __getitem__(self, idx):
        return create_sample_legacy(
            self.uniform.xs[idx], self.uniform.ys[idx], self.uniform.sample_ids[idx]
        )

    def __len__(self):
        return self.uniform.xs.shape[0]
