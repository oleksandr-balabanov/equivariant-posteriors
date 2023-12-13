#!/usr/bin/env python
import torch
import torchmetrics as tm

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainEval
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.metric import Metric
from lib.models.dense import DenseConfig
from lib.data_registry import DataSineConfig

from lib.train import load_or_create_state
from lib.train import do_training
from lib.ddp import ddp_setup


def main():
    device_id = ddp_setup("gloo")
    print(f"Using device {device_id}")
    loss = torch.nn.MSELoss()

    def mse_loss(outputs, batch):
        return loss(outputs["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=DenseConfig(d_hidden=100),
        train_data_config=DataSineConfig(
            input_shape=torch.Size([1]), output_shape=torch.Size([1])
        ),
        val_data_config=DataSineConfig(
            input_shape=torch.Size([1]), output_shape=torch.Size([1])
        ),
        loss=mse_loss,
        optimizer=OptimizerConfig(optimizer=torch.optim.Adam, kwargs=dict()),
        batch_size=2,
    )
    train_eval = TrainEval(
        train_metrics=[
            lambda: Metric(
                lambda output, batch: tm.functional.mean_absolute_error(
                    output["predictions"], batch["target"]
                )
            ),
            lambda: Metric(
                lambda output, batch: tm.functional.mean_squared_error(
                    output["predictions"], batch["target"]
                )
            ),
        ],
        validation_metrics=[
            lambda: Metric(
                lambda output, batch: tm.functional.mean_absolute_error(
                    output["predictions"], batch["target"]
                )
            ),
            lambda: Metric(
                lambda output, batch: tm.functional.mean_squared_error(
                    output["predictions"], batch["target"]
                )
            ),
        ],
    )
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=1),
        train_config=train_config,
        train_eval=train_eval,
        epochs=20,
        save_nth_epoch=5,
        validate_nth_epoch=5,
    )
    state = load_or_create_state(train_run, device_id)
    do_training(train_run, state, device_id)


if __name__ == "__main__":
    main()
