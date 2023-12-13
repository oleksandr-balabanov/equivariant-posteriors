#!/usr/bin/env python
import torch
from pathlib import Path

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.classification_metrics import create_classification_metrics
from lib.data_registry import DataSpiralsConfig
from lib.datasets.spiral_visualization import visualize_spiral
from lib.models.mlp import MLPClassConfig
from lib.generic_ablation import generic_ablation

from lib.distributed_trainer import distributed_train


def create_config(mlp_dim, ensemble_id):
    loss = torch.nn.CrossEntropyLoss()

    def ce_loss(output, batch):
        return loss(output["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=MLPClassConfig(widths=[mlp_dim, mlp_dim]),
        train_data_config=DataSpiralsConfig(seed=0, N=1000),
        val_data_config=DataSpiralsConfig(seed=1, N=500),
        loss=ce_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam, kwargs=dict(weight_decay=0.0001)
        ),
        batch_size=500,
        ensemble_id=ensemble_id,
    )
    train_eval = create_classification_metrics(visualize_spiral, 2)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=1),
        train_config=train_config,
        train_eval=train_eval,
        epochs=200,
        save_nth_epoch=20,
        validate_nth_epoch=20,
    )
    return train_run


if __name__ == "__main__":
    configs = generic_ablation(
        create_config,
        dict(mlp_dim=[200, 10, 50, 100, 200], ensemble_id=list(range(5))),
    )
    distributed_train(configs)
