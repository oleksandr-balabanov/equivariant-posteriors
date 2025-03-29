from typing import Dict
import random
import functools
import torch
import numpy as np
import torchvision
from dataclasses import dataclass
from lib.dataspec import DataSpec
from lib.serialization import serialize_human
from lib.data_utils import create_sample_legacy
from lib.train_dataclasses import TrainEpochState


@dataclass(frozen=True)
class DataCIFARConfig:
    validation: bool = False
    augment: bool = True

    def serialize_human(self):
        return serialize_human(self.__dict__)  # dict(validation=self.validation)


def cutout(x, c):
    x0, y0 = random.randint(0, 32 - 8), random.randint(0, 32 - 8)
    x[..., y0 : y0 + c, x0 : x0 + c] = 0.0
    return x


class DataCIFAROld(torch.utils.data.Dataset):
    def __init__(self, data_config: DataCIFARConfig):
        transform_stack = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
            ),
        ]
        if not data_config.validation and data_config.augment:
            transform_stack.append(
                torchvision.transforms.RandomCrop(32, padding=4, padding_mode="reflect")
            )
            transform_stack.append(torchvision.transforms.RandomHorizontalFlip())
            transform_stack.append(lambda x: cutout(x, 8))
        self.CIFAR = torchvision.datasets.CIFAR10(
            "datasets",
            train=not data_config.validation,
            download=True,
            transform=torchvision.transforms.Compose(transform_stack),
        )
        self.n_classes = 10
        self.config = data_config

    @staticmethod
    def data_spec(config: DataCIFARConfig):
        return DataSpec(
            input_shape=torch.Size([3, 32, 32]),
            target_shape=torch.Size([1]),
            output_shape=torch.Size([10]),
        )

    @functools.cache
    def __getitem__(self, idx):
        cifar_sample = self.CIFAR[idx]
        # image = torch.flatten(cifar_sample[0]).reshape(3, 32, 32)
        # image = image.unfold(0, 14, 14)
        # image = image.unfold(1, 14, 14)
        # image = image.reshape(2 * 2, 14 * 14)
        # image = image.reshape(-1, 16 * 16)
        return create_sample_legacy(cifar_sample[0], cifar_sample[1], idx)

    def __len__(self):
        return len(self.CIFAR)


class DataCIFAR(torchvision.datasets.CIFAR10):
    def __init__(self, data_config: DataCIFARConfig):
        transform_stack = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
            ),
        ]
        if not data_config.validation and data_config.augment:
            transform_stack.append(
                torchvision.transforms.RandomCrop(32, padding=4, padding_mode="reflect")
            )
            transform_stack.append(torchvision.transforms.RandomHorizontalFlip())
            transform_stack.append(lambda x: cutout(x, 8))
        super().__init__(
            "datasets",
            train=not data_config.validation,
            download=True,
            transform=torchvision.transforms.Compose(transform_stack),
        )
        self.n_classes = 10
        self.config = data_config

    def name(self):
        if self.config.validation:
            subset = "val"
        else:
            subset = "train"

        return f"CIFAR10_{subset}"

    @staticmethod
    def data_spec(config: DataCIFARConfig):
        return DataSpec(
            input_shape=torch.Size([3, 32, 32]),
            target_shape=torch.Size([1]),
            output_shape=torch.Size([10]),
        )

    @staticmethod
    def sample_id_spec(config):
        return ["idx"]

    # @functools.cache
    def __getitem__(self, idx):
        cifar_sample = super().__getitem__(idx)  # self.CIFAR[idx]
        # image = torch.flatten(cifar_sample[0]).reshape(3, 32, 32)
        # image = image.unfold(0, 14, 14)
        # image = image.unfold(1, 14, 14)
        # image = image.reshape(2 * 2, 14 * 14)
        # image = image.reshape(-1, 16 * 16)
        return create_sample_legacy(cifar_sample[0], cifar_sample[1], np.array([idx]))

    # def __len__(self):
    # return len(self.CIFAR)
