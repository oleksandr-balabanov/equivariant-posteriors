import torch
from typing import List, Optional
from lib.data_factory import get_factory
from dataclasses import dataclass, field
from lib.data_utils import create_sample_legacy
import math


@dataclass(frozen=True)
class DataSubsetConfig:
    data_config: object
    subset: List[int] = field(
        default_factory=lambda: []
    )  # The subset to use, one of ``all`` or the keys in ``cifarc_subsets``
    minimum_epoch_length: Optional[int] = None

    def serialize_human(self):
        return dict(data_config=self.data_config.serialize_human(), subset=self.subset)


# fmt:on
class DataSubset:
    def __init__(self, config: DataSubsetConfig):
        assert isinstance(config.subset, list)
        self.ds = get_factory().create(config.data_config)
        if config.minimum_epoch_length is not None:
            n_cycles = math.ceil(config.minimum_epoch_length / len(config.subset))
            self.subset = list(config.subset * n_cycles)
        else:
            self.subset = list(config.subset)
        self.n_classes = self.ds.n_classes
        self.config = config

    @staticmethod
    def data_spec(config: DataSubsetConfig):
        return get_factory().get_class(config.data_config).data_spec(config.data_config)

    @staticmethod
    def sample_id_spec(config: DataSubsetConfig):
        return ["idx"]
        # return get_factory().get_class(config.data_config).sample_id_spec(config.data_config)

    def __getitem__(self, idx):
        # breakpoint()
        x, y, sample_id = self.ds[self.subset[idx]]
        return create_sample_legacy(x, y, torch.tensor([idx]))

    def __len__(self):
        return len(self.subset)
