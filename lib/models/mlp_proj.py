import math
import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec
from typing import List
import lib.model_factory as model_factory

# from lib.model_factory import get_factory

# from lib.models.mlp import MLPClass, MLPClassConfig


@dataclass(frozen=True)
class MLPProjClassConfig:
    # mlp_config: MLPClassConfig
    model_config: object
    n_proj: int

    def serialize_human(self):
        return self.__dict__


class MLPProjClass(torch.nn.Module):
    def __init__(self, config: MLPProjClassConfig, data_spec: DataSpec):
        super().__init__()
        self.config = config
        factory = model_factory.get_factory()
        self.model = factory.create(
            config.model_config, data_spec
        )  # MLPClass(config.mlp_config, data_spec)
        self.mlp_proj = torch.nn.Linear(
            data_spec.output_shape[-1], config.n_proj, bias=True
        )
        self.mlp_out = torch.nn.Linear(
            config.n_proj, data_spec.output_shape[-1], bias=True
        )

        torch.nn.init.normal_(self.mlp_proj.weight, 0.0, math.sqrt(1.0 / 10.0))
        torch.nn.init.normal_(self.mlp_proj.bias, 0.0, std=math.sqrt(1e-7))

        torch.nn.init.normal_(self.mlp_out.weight, 0.0, math.sqrt(1.0 / 2.0))
        torch.nn.init.normal_(self.mlp_out.bias, 0.0, std=math.sqrt(1e-7))

    def forward(self, batch):
        x = batch["input"]
        out = self.model(x)
        projection = self.mlp_proj(out["logits"])
        logits = self.mlp_out(projection)
        return dict(
            projection=projection,
            logits=logits,
            predictions=torch.softmax(logits.detach(), dim=-1),
            base_model_logits=out["logits"],
            base_model_predictions=torch.softmax(out["logits"].detach(), dim=-1),
        )
