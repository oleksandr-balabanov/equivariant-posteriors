#!/usr/bin/env python
import torch
import torchmetrics as tm
from pathlib import Path

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
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


def create_config(embed_d, ensemble_id):
    ce_loss = torch.nn.CrossEntropyLoss()

    def loss(outputs, batch):
        return ce_loss(outputs["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=TransformerConfig(
            embed_d=embed_d,
            mlp_dim=10,
            n_seq=2,
            batch_size=500,
            num_layers=2,
            num_heads=1,
        ),
        train_data_config=DataSpiralsConfig(seed=0, N=1000),
        val_data_config=DataSpiralsConfig(seed=1, N=500),
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam, kwargs=dict(weight_decay=0.0001)
        ),
        loss=loss,
        batch_size=500,
        ensemble_id=ensemble_id,
    )
    train_eval = create_classification_metrics(visualize_spiral, 2)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=1),
        train_config=train_config,
        train_eval=train_eval,
        epochs=500,
        save_nth_epoch=20,
        validate_nth_epoch=20,
    )
    return train_run


def create_values():
    return dict(embed_d=[100, 1, 5, 20, 50, 100], ensemble_id=list(range(5)))


if __name__ == "__main__":
    configs = generic_ablation(create_config, create_values())
    distributed_train(configs)
