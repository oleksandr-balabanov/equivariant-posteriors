import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec


@dataclass(frozen=True)
class ConvLAPConfig:
    def serialize_human(self):
        return self.__dict__


class ConvLAP(torch.nn.Module):
    def __init__(self, config: ConvLAPConfig, data_spec: DataSpec):
        super().__init__()
        self.config = config
        self.c1 = torch.nn.Conv2d(3, 96, kernel_size=5)
        self.c2 = torch.nn.Conv2d(96, 192, kernel_size=5)
        self.c3 = torch.nn.Conv2d(192, 192, kernel_size=3)
        self.c4 = torch.nn.Conv2d(192, 192, kernel_size=1)
        self.c5 = torch.nn.Conv2d(192, data_spec.output_shape[-1], kernel_size=1)

        self.bn1 = torch.nn.BatchNorm2d(96, momentum=0.99, eps=1e-3)
        self.bn2 = torch.nn.BatchNorm2d(192, momentum=0.99, eps=1e-3)
        self.bn3 = torch.nn.BatchNorm2d(192, momentum=0.99, eps=1e-3)
        self.bn4 = torch.nn.BatchNorm2d(192, momentum=0.99, eps=1e-3)

    def forward(self, batch):
        x = batch["input"]
        x = x.reshape(-1, 3, 32, 32)
        x = self.c1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)

        x = self.c2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)

        x = self.c3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)

        x = self.c4(x)
        x = self.bn4(x)
        x = torch.nn.functional.relu(x)

        x = self.c5(x)
        # x = self.bn5(x)
        x = torch.nn.functional.relu(x)

        # breakpoint()
        logits = torch.nn.functional.avg_pool2d(x, 2)
        logits = logits.squeeze()
        return dict(logits=logits, predictions=torch.softmax(logits.detach(), dim=-1))
