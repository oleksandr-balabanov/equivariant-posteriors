from typing import Dict
import functools
import torch
import torchvision
from dataclasses import dataclass
from lib.dataspec import DataSpec
from lib.data_utils import create_sample_legacy
from lib.train_dataclasses import TrainEpochState


@dataclass(frozen=True)
class DataCIFAR2Config:
    validation: bool = False

    def serialize_human(self):
        return dict(validation=self.validation)


class DataCIFAR2(torch.utils.data.Dataset):
    def __init__(self, data_config: DataCIFAR2Config):
        self.CIFAR = torchvision.datasets.CIFAR10(
            "datasets",
            train=not data_config.validation,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),
                ]
            ),
        )
        self.n_classes = 2
        self.class_names_to_idx = {
            "airplane": 0,
            "automobile": 0,
            "bird": 1,
            "cat": 1,
            "deer": 1,
            "dog": 1,
            "frog": 1,
            "horse": 1,
            "ship": 0,
            "truck": 0,
        }
        self.class_map = {
            self.CIFAR.class_to_idx[class_name]: self.class_names_to_idx[class_name]
            for class_name in self.class_names_to_idx.keys()
        }
        self.config = data_config

    @staticmethod
    def data_spec():
        return DataSpec(
            input_shape=torch.Size([3, 32, 32]),
            target_shape=torch.Size([1]),
            output_shape=torch.Size([2]),
        )

    @functools.cache
    def __getitem__(self, idx):
        cifar_sample = self.CIFAR[idx]
        # image = torch.flatten(cifar_sample[0]).reshape(3, 32, 32)
        # image = image.unfold(0, 14, 14)
        # image = image.unfold(1, 14, 14)
        # image = image.reshape(2 * 2, 14 * 14)
        # image = image.reshape(-1, 16 * 16)
        class_idx = self.class_map[cifar_sample[1]]
        return create_sample_legacy(cifar_sample[0], class_idx, idx)

    def __len__(self):
        return len(self.CIFAR)
