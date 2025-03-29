from typing import Dict
import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec
from lib.data_utils import create_sample_legacy
from lib.train_dataclasses import TrainEpochState


@dataclass(frozen=True)
class DataSpiralsConfig:
    seed: int
    N: int
    angle_factor: float = 1.0

    def serialize_human(self):
        return dict(seed=self.seed, N=self.N)


@dataclass
class Spiral:
    xs: torch.Tensor
    ys: torch.Tensor
    sample_ids: torch.Tensor


def generate_spiral_points(N, angle_factor):
    angles = 4 * 3 * torch.rand(N, 1)
    r = 1.0 + 0.1 * torch.randn(N, 1)

    xs1 = torch.stack(
        [
            r * angles / (4 * 3) * torch.cos(angle_factor * angles),
            r * angles / (4 * 3) * torch.sin(angle_factor * angles),
        ],
        dim=1,
    )
    ys1 = torch.zeros(N, dtype=torch.long)

    xs2 = torch.stack(
        [
            r * angles / (4 * 3) * torch.cos(angle_factor * angles + 3.14),
            r * angles / (4 * 3) * torch.sin(angle_factor * angles + 3.14),
        ],
        dim=1,
    )
    ys2 = torch.ones(N, dtype=torch.long)
    xs = torch.concat([xs1, xs2], dim=0)
    ys = torch.concat([ys1, ys2], dim=0)
    sample_ids = torch.arange(0, xs.shape[0], 1, dtype=torch.int32)
    return Spiral(xs, ys, sample_ids)


class DataSpirals(torch.utils.data.Dataset):
    def __init__(self, data_config: DataSpiralsConfig):
        torch.manual_seed(data_config.seed)
        self.spiral = generate_spiral_points(data_config.N, data_config.angle_factor)
        self.n_classes = 2

    @staticmethod
    def data_spec(config):
        return DataSpec(
            input_shape=torch.Size([2, 1]),
            target_shape=torch.Size([1]),
            output_shape=torch.Size([2]),
        )

    def __getitem__(self, idx):
        return create_sample_legacy(
            self.spiral.xs[idx], self.spiral.ys[idx], self.spiral.sample_ids[idx]
        )

    def __len__(self):
        return self.spiral.xs.shape[0]
