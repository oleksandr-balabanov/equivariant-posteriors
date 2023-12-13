#!/usr/bin/env python
import torch
import torchmetrics as tm
from pathlib import Path

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import ComputeConfig
from lib.train_dataclasses import OptimizerConfig
from lib.models.transformer import TransformerConfig
from lib.data_registry import DataSpiralsConfig
from lib.datasets.spiral_visualization import visualize_spiral
from lib.generic_ablation import generic_ablation
from lib.classification_metrics import create_classification_metrics
from lib.distributed_trainer import distributed_train


def loss(preds, target):
    return torch.nn.functional.binary_cross_entropy(preds, target, reduction="none")


def bce(preds, target):
    return tm.functional.classification.binary_calibration_error(
        preds, target, n_bins=15
    )


def create_config(mlp_dim, ensemble_id):
    ce_loss = torch.nn.CrossEntropyLoss()

    def loss(outputs, batch):
        return ce_loss(outputs["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=TransformerConfig(
            embed_d=50,
            mlp_dim=mlp_dim,
            n_seq=2,
            batch_size=500,
            num_layers=1,
            num_heads=1,
        ),
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam, kwargs=dict(weight_decay=0.0001)
        ),
        train_data_config=DataSpiralsConfig(seed=0, N=1000),
        val_data_config=DataSpiralsConfig(seed=1, N=500),
        loss=loss,
        batch_size=500,
        ensemble_id=ensemble_id,
    )
    train_eval = create_classification_metrics(
        data_visualizer=visualize_spiral, n_classes=2
    )
    train_run = TrainRun(
        compute_config=ComputeConfig(False, 0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=500,
        save_nth_epoch=20,
        validate_nth_epoch=20,
    )
    return train_run


if __name__ == "__main__":
    configs = generic_ablation(
        create_config,
        dict(mlp_dim=[1, 5, 10, 20, 50], ensemble_id=list(range(5))),
    )
    distributed_train(configs)
