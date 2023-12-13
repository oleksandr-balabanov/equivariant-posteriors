import torch
from dataclasses import dataclass, field
from typing import List
from typing import Callable
from typing import Union
from typing import Set

from lib.metric import Metric
from lib.timing_metric import Timing
from lib.stable_hash import stable_hash
import lib.serialize_human


@dataclass
class TrainEpochState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_metrics: List[Metric]
    validation_metrics: List[Metric]
    train_dataloader: torch.utils.data.DataLoader
    timing_metric: Timing
    epoch: int
    batch: int
    next_visualization: float
    next_visualizer: int
    val_dataloader: torch.utils.data.DataLoader = None
    device_memory_stats: dict = None
    host_memory_stats: object = None
    psql_query_cache: Set = field(default_factory=lambda: set())


@dataclass
class TrainEpochSpec:
    loss: object
    device_id: int


@dataclass
class OptimizerConfig:
    optimizer: torch.optim.Optimizer
    kwargs: dict


@dataclass
class TrainConfig:
    model_config: object
    train_data_config: object
    loss: torch.nn.Module
    optimizer: OptimizerConfig
    batch_size: int
    gradient_clipping: Union[None, float] = None
    ensemble_id: int = 0
    _version: int = 0
    val_data_config: object = None
    post_model_create_hook: object = None
    model_pre_train_hook: object = None
    extra: object = None

    def serialize_human(self):
        import lib.model_factory as model_factory
        import lib.data_factory as data_factory

        val_data = None
        if self.val_data_config is not None:
            val_data = dict(
                config=lib.serialize_human.serialize_human(self.val_data_config),
                name=data_factory.get_factory()
                .get_class(self.val_data_config)
                .__name__,
            )
        return dict(
            model=dict(
                config=self.model_config.serialize_human(),
                name=model_factory.get_factory().get_class(self.model_config).__name__,
                post_model_create_hook=self.post_model_create_hook.__name__
                if self.post_model_create_hook is not None
                else None,
                model_pre_train_hook=self.model_pre_train_hook.__name__
                if self.model_pre_train_hook is not None
                else None,
            ),
            data=dict(
                config=self.train_data_config.serialize_human(),
                name=data_factory.get_factory()
                .get_class(self.train_data_config)
                .__name__,
            ),
            val_data=val_data,
            loss=self.loss.__class__.__name__,
            optimizer=dict(
                name=self.optimizer.optimizer.__name__,
                kwargs=self.optimizer.kwargs,
            ),
            batch_size=self.batch_size,
            ensemble_id=self.ensemble_id,
            extra=lib.serialize_human.serialize_human(self.extra),
        )

    def ensemble_dict(self):
        ensemble_config = self.__dict__.copy()
        ensemble_config.pop("ensemble_id")
        return ensemble_config


@dataclass
class TrainEval:
    train_metrics: List[Callable[[], Metric]]
    validation_metrics: List[Callable[[], Metric]]
    data_visualizer: Callable[[object, TrainEpochState], None] = None

    def serialize_human(self):
        return dict(
            train_metrics=[metric().name() for metric in self.train_metrics],
            val_metric=[metric().name() for metric in self.validation_metrics],
        )


@dataclass
class ComputeConfig:
    distributed: bool
    num_workers: int

    def serialize_human(self):
        return self.__dict__


@dataclass
class TrainRun:
    compute_config: ComputeConfig
    train_config: TrainConfig
    train_eval: TrainEval
    epochs: int
    save_nth_epoch: int
    validate_nth_epoch: int

    def serialize_human(self):
        return dict(
            compute_config=self.compute_config.serialize_human(),
            train_config=self.train_config.serialize_human(),
            train_id=stable_hash(self.train_config),
            ensemble_id=stable_hash(self.train_config.ensemble_dict()),
            train_eval=self.train_eval.serialize_human(),
            epochs=self.epochs,
            save_nth_epoch=self.save_nth_epoch,
        )


@dataclass
class EnsembleConfig:
    members: list[TrainRun]

    def serialize_human(self):
        return [member_config.serialize_human() for member_config in self.members]


@dataclass
class Ensemble:
    member_configs: list[TrainRun]
    members: list[torch.nn.Module]
    n_members: int
