from typing import Dict
import functools
import torch
import torchvision
from dataclasses import dataclass
from lib.dataspec import DataSpec
from lib.data_utils import create_sample_legacy
from lib.train_dataclasses import TrainEpochState


@dataclass(frozen=True)
class DataMNISTConfig:
    validation: bool = False

    def serialize_human(self):
        return dict(validation=self.validation)


class DataMNIST(torch.utils.data.Dataset):
    def __init__(self, data_config: DataMNISTConfig):
        self.MNIST = torchvision.datasets.MNIST(
            "datasets",
            train=not data_config.validation,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        self.n_classes = 10
        self.config = data_config

    @staticmethod
    def data_spec(config):
        return DataSpec(
            input_shape=torch.Size([4, 14 * 14]),
            target_shape=torch.Size([1]),
            output_shape=torch.Size([10]),
        )

    @functools.cache
    def __getitem__(self, idx):
        mnist_sample = self.MNIST[idx]
        image = torch.flatten(mnist_sample[0]).reshape(28, 28)
        # image = image.unfold(0, 14, 14)
        # image = image.unfold(1, 14, 14)
        # image = image.reshape(2 * 2, 14 * 14)
        image = image.reshape(-1, 14 * 14)
        return create_sample_legacy(image, mnist_sample[1], idx)

    def __len__(self):
        return len(self.MNIST)
