import traceback
from pathlib import Path
import torch
import logging
import time
from typing import Dict

from filelock import FileLock
from lib.metric import MetricSample, Metric
from lib.timing_metric import Timing
import lib.data_factory as data_factory
from lib.data_utils import get_sampler
import lib.model_factory as model_factory
from lib.render_dataframe import render_dataframe
from lib.render_psql import render_psql

from lib.train_dataclasses import TrainEpochState
from lib.train_dataclasses import TrainEpochSpec
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import ComputeConfig

from lib.serialization import SerializeConfig
from lib.serialization import DeserializeConfig
from lib.serialization import deserialize
from lib.serialization import serialize
from lib.serialization import write_status_file

from lib.train_visualization import visualize_progress
from lib.train_visualization import visualize_progress_batches

from lib.paths import get_checkpoint_path, get_lock_path
import lib.ddp as ddp


def evaluate_metrics_on_data(
    model,
    metrics: Dict[str, Metric],
    data_config,
    batch_size: int,
    compute_config: ComputeConfig,
    device,
):
    dataloader = create_dataloader(
        data_config, batch_size, shuffle=False, compute_config=compute_config
    )
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(batch)

            metric_sample = MetricSample(
                output=output,
                batch=batch,
                # prediction=output["predictions"],
                epoch=-1,
            )
            for metric_name, metric in metrics.items():
                metric(metric_sample)


def validate(
    train_epoch_state: TrainEpochState,
    train_epoch_spec: TrainEpochSpec,
    train_run: TrainRun,
):
    if train_epoch_state.val_dataloader is None:
        return
    if train_epoch_state.epoch % train_run.validate_nth_epoch != 0:
        # Evaluate if this is the last epoch regardless
        if train_epoch_state.epoch != train_run.epochs:
            return
    model = train_epoch_state.model
    dataloader = train_epoch_state.val_dataloader
    device = train_epoch_spec.device_id

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(batch)
            # metric_sample = dataloader.dataset.create_metric_sample(
            #     output=output, batch=batch, train_epoch_state=train_epoch_state
            # )
            metric_sample = MetricSample(
                output=output, batch=batch, epoch=train_epoch_state.epoch
            )
            for metric in train_epoch_state.validation_metrics:
                metric(metric_sample)


def train(
    train_run: TrainRun,
    train_epoch_state: TrainEpochState,
    train_epoch_spec: TrainEpochSpec,
):
    model = train_epoch_state.model
    dataloader = train_epoch_state.train_dataloader
    loss = train_epoch_spec.loss
    optimizer = train_epoch_state.optimizer
    device = train_epoch_spec.device_id

    model.train()

    if train_run.train_config.model_pre_train_hook is not None:
        model = train_run.train_config.model_pre_train_hook(model, train_run=train_run)

    if dataloader.sampler.__class__.__name__ == "DistributedSampler":
        dataloader.sampler.set_epoch(train_epoch_state.epoch)

    visualizers = [visualize_progress, visualize_progress_batches]
    # train_epoch_state.batch = 0
    for i, batch in enumerate(dataloader):
        train_epoch_state.batch += 1
        train_epoch_state.timing_metric.start("to_device")
        batch = {k: v.to(device) for k, v in batch.items()}
        train_epoch_state.timing_metric.stop("to_device")
        train_epoch_state.timing_metric.start("model_forward")
        output = model(batch)
        train_epoch_state.timing_metric.stop("model_forward")

        train_epoch_state.timing_metric.start("loss_opt_step")
        loss_val = loss(output, batch)
        optimizer.zero_grad()
        loss_val.backward()
        if train_run.train_config.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_run.train_config.gradient_clipping
            )
        optimizer.step()
        train_epoch_state.timing_metric.stop("loss_opt_step")

        if i == 0 and torch.cuda.is_available():
            train_epoch_state.device_memory_stats = (
                torch.cuda.memory_stats_as_nested_dict()
            )

        train_epoch_state.timing_metric.start("train_metrics")
        metric_sample = MetricSample(
            output=output, batch=batch, epoch=train_epoch_state.epoch
        )
        for metric in train_epoch_state.train_metrics:
            metric(metric_sample)
        train_epoch_state.timing_metric.stop("train_metrics")

        try:
            now = time.time()
            if now > train_epoch_state.next_visualization and ddp.get_rank() == 0:
                write_status_file(
                    SerializeConfig(
                        train_run=train_run, train_epoch_state=train_epoch_state
                    )
                )
                train_epoch_state.timing_metric.start("psql")
                last_postgres_result = render_psql(train_run, train_epoch_state)
                train_epoch_state.timing_metric.stop("psql")
                train_epoch_state.next_visualization = now + 5
                train_epoch_state.next_visualizer = (
                    train_epoch_state.next_visualizer + 1
                ) % 2
                train_epoch_state.timing_metric.start("visualize")
                visualizers[train_epoch_state.next_visualizer](
                    train_epoch_state,
                    train_run,
                    last_postgres_result,
                    train_epoch_spec.device_id,
                )
                train_epoch_state.timing_metric.stop("visualize")
        except Exception as e:
            logging.error("Visualization failed")
            logging.error(str(e))
            print(traceback.format_exc())


def create_dataloader(
    data_config, batch_size: int, shuffle: bool, compute_config: ComputeConfig
):
    ds = data_factory.get_factory().create(data_config)
    sampler, shuffle = get_sampler(compute_config, ds, shuffle=shuffle)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        sampler=sampler,
        num_workers=compute_config.num_workers,
        collate_fn=ds.collate_fn if hasattr(ds, "collate_fn") else None,
    )
    return dataloader


def create_initial_state(train_run: TrainRun, device_id):
    train_config = train_run.train_config

    train_ds = data_factory.get_factory().create(train_config.train_data_config)

    train_sampler, train_shuffle = get_sampler(
        train_run.compute_config, train_ds, shuffle=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        drop_last=False,
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=train_run.compute_config.num_workers,
        collate_fn=train_ds.collate_fn if hasattr(train_ds, "collate_fn") else None,
    )
    if train_config.val_data_config is not None:
        val_ds = data_factory.get_factory().create(train_config.val_data_config)
        val_sampler, val_shuffle = get_sampler(
            train_run.compute_config, val_ds, shuffle=False
        )
        assert val_shuffle is False
        val_dataloader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=train_config.batch_size,
            shuffle=val_shuffle,
            drop_last=False,
            sampler=val_sampler,
            num_workers=train_run.compute_config.num_workers,
            collate_fn=val_ds.collate_fn if hasattr(val_ds, "collate_fn") else None,
        )
    else:
        val_dataloader = None

    torch.manual_seed(train_config.ensemble_id)
    init_model = model_factory.get_factory().create(
        train_config.model_config,
        train_ds.__class__.data_spec(train_config.train_data_config),
    )

    if train_config.post_model_create_hook is not None:
        init_model = train_config.post_model_create_hook(
            init_model, train_run=train_run
        )

    init_model = init_model.to(torch.device(device_id))
    if train_run.compute_config.distributed:
        device_id_list = [device_id]
        init_model = torch.nn.parallel.DistributedDataParallel(
            init_model, device_ids=device_id_list, find_unused_parameters=True
        )

    opt = train_config.optimizer.optimizer(
        init_model.parameters(), **train_config.optimizer.kwargs
    )
    train_metrics = [metric() for metric in train_run.train_eval.train_metrics]
    validation_metrics = [
        metric() for metric in train_run.train_eval.validation_metrics
    ]

    return TrainEpochState(
        model=init_model,
        optimizer=opt,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        epoch=0,
        batch=0,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        next_visualization=0.0,
        next_visualizer=0,
        timing_metric=Timing(),
    )


def load_or_create_state(train_run: TrainRun, device_id):
    config = DeserializeConfig(
        train_run=train_run,
        device_id=device_id,
    )

    state = None
    try:
        state = deserialize(config)
    except Exception as e:
        print("ERROR: Failed to load checkpoint, creating a new initial state.")
        raise e
        # print(str(e))

    if state is None:
        state = create_initial_state(
            train_run=config.train_run,
            device_id=config.device_id,
        )

    return state


def do_training(train_run: TrainRun, state: TrainEpochState, device_id):
    train_epoch_spec = TrainEpochSpec(
        loss=train_run.train_config.loss,
        device_id=device_id,
    )

    print("Serializing config...")
    serialize_config = SerializeConfig(train_run=train_run, train_epoch_state=state)

    print("Run epochs...")
    checkpoint_path = get_lock_path(train_run.train_config)
    lock = FileLock(f"{checkpoint_path}", 1)
    with lock:
        while state.epoch < train_run.epochs:
            train(train_run, state, train_epoch_spec)
            validate(state, train_epoch_spec, train_run)
            state.epoch += 1
            serialize(serialize_config)
            # last_postgres_result = render_psql(train_run, state)
            # try:
            #     now = time.time()
            #     if now > next_visualization:
            #         next_visualization = now + 1
            #         if ddp.get_rank() == 0:
            #             visualize_progress(
            #                 state, train_run, last_postgres_result, device_id
            #             )
            # except Exception as e:
            #     logging.error("Visualization failed")
            #     logging.error(str(e))
            #     print(traceback.format_exc())

        print("Render dataframe...")
        df = render_dataframe(train_run, state)

        print("Render psql...")
        render_psql(train_run, state, block=True)

        print("Pickling dataframe...")
        df_path = get_checkpoint_path(train_run.train_config) / "df.pickle"
        if not Path(df_path).is_file():
            df.to_pickle(path=df_path)

    print("Done.")
