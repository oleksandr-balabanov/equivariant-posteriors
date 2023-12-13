from typing import List
from lib.data_factory import get_factory
from lib.data_utils import create_sample_legacy
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class DataJoinConfig:
    data_configs: List[object]

    def serialize_human(self):
        return [config.serialize_human() for config in self.data_configs]


# fmt:on
class DataJoin:
    def __init__(self, config: DataJoinConfig):
        self.dss = [
            get_factory().create(part_config) for part_config in config.data_configs
        ]
        self.lens = [len(ds) for ds in self.dss]

        self.index = []
        for ds_idx, ds in enumerate(self.dss):
            local_idxs = list(range(len(ds)))
            global_idxs = [(ds_idx, local_idx) for local_idx in local_idxs]
            self.index = self.index + global_idxs
        self.config = config

    @staticmethod
    def data_spec(config: DataJoinConfig):
        return (
            get_factory()
            .get_class(config.data_configs[0])
            .data_spec(config.data_configs[0])
        )

    @staticmethod
    def sample_id_spec(_config):
        return ["idx"]

    def __getitem__(self, idx):
        ds_idx, sample_idx = self.index[idx]
        x, target, sample_ids = self.dss[ds_idx][sample_idx]
        # breakpoint()
        return create_sample_legacy(x, target, torch.tensor([idx]))

    def __len__(self):
        return len(self.index)
