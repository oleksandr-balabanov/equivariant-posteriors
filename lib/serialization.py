import torch
from dataclasses import dataclass
from typing import List
from datetime import datetime

from lib.train_dataclasses import TrainEpochState
from lib.train_dataclasses import TrainRun
import lib.data_factory as data_factory
from lib.metric import Metric
from lib.timing_metric import Timing
from lib.paths import get_checkpoint_path, get_or_create_checkpoint_path
import lib.model_factory as model_factory
from lib.data_utils import get_sampler
from lib.ddp import get_rank
from lib.stable_hash import json_dumps_dataclass_str
import shutil
from lib.serialize_human import serialize_human


@dataclass
class SerializeConfig:
    train_run: TrainRun
    train_epoch_state: TrainEpochState


def serialize_metrics(metrics: List[Metric]):
    serialized_metrics = [(metric.name(), metric.serialize()) for metric in metrics]
    serialized_metrics = {name: value for name, value in serialized_metrics}
    return serialized_metrics


@dataclass
class FileStructure:
    model: object = None
    optimizer: object = None
    epoch: object = None
    train_metrics: object = None
    validation_metrics: object = None
    train_run: object = None
    timing_metric: object = None


def write_status_file(config: SerializeConfig):
    if get_rank() != 0:
        # print("I am not rank 0 so not serializing...")
        return

    checkpoint_path = get_or_create_checkpoint_path(config.train_run.train_config)
    with open(checkpoint_path / "status_tmp", "w") as status_file:
        status_file.write(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Epoch {config.train_epoch_state.epoch} / {config.train_run.epochs}, Batch {config.train_epoch_state.batch} / {config.train_run.epochs * len(config.train_epoch_state.train_dataloader)}"
        )
    shutil.move(checkpoint_path / "status_tmp", checkpoint_path / "status")


def serialize(config: SerializeConfig):
    if get_rank() != 0:
        # print("I am not rank 0 so not serializing...")
        return
    train_config = config.train_run.train_config
    train_epoch_state = config.train_epoch_state
    train_run = config.train_run

    if train_epoch_state.epoch % train_run.save_nth_epoch != 0:
        # Save if this is the last epoch regardless
        if train_epoch_state.epoch != train_run.epochs:
            return
    file_data = FileStructure(
        model=train_epoch_state.model.state_dict(),
        optimizer=train_epoch_state.optimizer.state_dict(),
        epoch=train_epoch_state.epoch,
        train_metrics=serialize_metrics(train_epoch_state.train_metrics),
        validation_metrics=serialize_metrics(train_epoch_state.validation_metrics),
        train_run=config.train_run.serialize_human(),
        timing_metric=train_epoch_state.timing_metric.serialize(),
    )

    checkpoint_path = get_or_create_checkpoint_path(train_config)
    for key, value in file_data.__dict__.items():
        torch.save(value, checkpoint_path / f"{key}_tmp")
        shutil.move(checkpoint_path / f"{key}_tmp", checkpoint_path / key)

    with open(checkpoint_path / "train_run.json_tmp", "w") as train_run_json_file:
        train_run_json_file.write(json_dumps_dataclass_str(config.train_run, indent=2))
    shutil.move(
        checkpoint_path / "train_run.json_tmp", checkpoint_path / "train_run.json"
    )
    write_status_file(config)


def get_train_run_status(train_run: TrainRun):
    status_file = get_checkpoint_path(train_run.train_config) / "status"
    if status_file.is_file():
        try:
            return open(status_file).read()
        except:
            return "Couldn't read status..."
    else:
        return "No status file."


@dataclass
class DeserializeConfig:
    train_run: TrainRun
    device_id: torch.device


def is_serialized(train_run: TrainRun):
    train_config = train_run.train_config
    checkpoint_path = get_checkpoint_path(train_config) / "model"
    return checkpoint_path.is_file()


def create_model(config: DeserializeConfig, state_dict: torch.Tensor):
    train_config = config.train_run.train_config
    data_spec = (
        data_factory.get_factory()
        .get_class(train_config.train_data_config)
        .data_spec(train_config.train_data_config)
    )
    model = model_factory.get_factory().create(train_config.model_config, data_spec)
    model = model.to(torch.device(config.device_id))

    if config.train_run.compute_config.distributed:
        device_id_list = [config.device_id]
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=device_id_list, find_unused_parameters=True
        )

    model.load_state_dict(state_dict)

    if train_config.model_pre_train_hook is not None:
        model = train_config.model_pre_train_hook(model, config.train_run)

    return model


@dataclass
class DeserializedModel:
    model: torch.nn.Module
    epoch: int


def deserialize_model(config: DeserializeConfig):
    train_config = config.train_run.train_config
    checkpoint_path = get_checkpoint_path(train_config)
    if (
        not (checkpoint_path / "model").is_file()
        or not (checkpoint_path / "epoch").is_file()
    ):
        return None
    else:
        print(f"{checkpoint_path}")

    try:
        model_state_dict = torch.load(
            checkpoint_path / "model", map_location=torch.device(config.device_id)
        )

        model_epoch = torch.load(checkpoint_path / "epoch")
    except Exception as e:
        print(f"Failed to deserialize_model: {e}")
        return None

    return DeserializedModel(
        model=create_model(config, model_state_dict), epoch=model_epoch
    )


def deserialize(config: DeserializeConfig):
    train_config = config.train_run.train_config
    checkpoint_path = get_checkpoint_path(train_config)
    if not (checkpoint_path / "model").is_file():
        return None
    else:
        print(f"{checkpoint_path}")

    file_data = FileStructure()
    for key in list(file_data.__dict__.keys()):
        print(checkpoint_path / key)
        try:
            setattr(
                file_data,
                key,
                torch.load(
                    checkpoint_path / key, map_location=torch.device(config.device_id)
                ),
            )
        except RuntimeError as e:
            print(f"Deserialize failed: {e}")
            return None
        except EOFError as e:
            print(f"Deserialize failed: {e}")
            return None
        except KeyError as e:
            print(f"Deserialize failed: {e}")
            return None
        except Exception as e:
            print(f"Deserialize failed: {e}")
            return None
    data_dict = file_data.__dict__

    train_ds = data_factory.get_factory().create(train_config.train_data_config)

    train_sampler, train_shuffle = get_sampler(
        config.train_run.compute_config, train_ds, shuffle=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        drop_last=False,
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=config.train_run.compute_config.num_workers,
        collate_fn=train_ds.collate_fn if hasattr(train_ds, "collate_fn") else None,
    )
    if train_config.val_data_config is not None:
        val_ds = data_factory.get_factory().create(train_config.val_data_config)
        val_sampler, val_shuffle = get_sampler(
            config.train_run.compute_config, val_ds, shuffle=False
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=train_config.batch_size,
            shuffle=False,
            drop_last=False,
            sampler=val_sampler,
            num_workers=config.train_run.compute_config.num_workers,
            collate_fn=val_ds.collate_fn if hasattr(val_ds, "collate_fn") else None,
        )
    else:
        val_dataloader = None

    model = create_model(config, data_dict["model"])
    # model = model_factory.get_factory().create(
    #     train_config.model_config, val_ds.data_spec()
    # )
    # model = model.to(torch.device(config.device_id))

    # if config.train_run.compute_config.distributed:
    #     device_id_list = [config.device_id]
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=device_id_list, find_unused_parameters=True
    #     )

    # model.load_state_dict(data_dict["model"])

    # if train_config.model_pre_train_hook is not None:
    #     model = train_config.model_pre_train_hook(model, config.train_run)

    # if train_config.post_model_create_hook is not None:
    # model = train_config.post_model_create_hook(model, train_run=config.train_run)

    optimizer = train_config.optimizer.optimizer(
        model.parameters(), **train_config.optimizer.kwargs
    )
    optimizer.load_state_dict(data_dict["optimizer"])

    train_metrics = []
    for metric in config.train_run.train_eval.train_metrics:
        metric_instance = metric()
        metric_instance.deserialize(data_dict["train_metrics"][metric_instance.name()])
        train_metrics.append(metric_instance)

    validation_metrics = []
    for metric in config.train_run.train_eval.validation_metrics:
        metric_instance = metric()
        metric_instance.deserialize(
            data_dict["validation_metrics"][metric_instance.name()]
        )
        validation_metrics.append(metric_instance)

    timing_metric = Timing()
    timing_metric.deserialize(data_dict["timing_metric"])

    epoch = data_dict["epoch"]

    return TrainEpochState(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        batch=0,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        next_visualization=0.0,
        next_visualizer=0,
        timing_metric=timing_metric,
    )
