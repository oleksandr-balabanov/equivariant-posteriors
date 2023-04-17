#!/usr/bin/env python
import torch
import torchmetrics as tm
from pathlib import Path

from lib.train import TrainConfig
from lib.train import TrainEval
from lib.train import TrainRun
from lib.train import OptimizerConfig
from lib.metric import Metric
from lib.models.transformer import TransformerConfig
from lib.data import DataSpiralsConfig
from lib.ablation import ablation


def loss(preds, target):
    return torch.nn.functional.binary_cross_entropy(preds, target, reduction="none")


def bce(preds, target):
    return tm.functional.classification.binary_calibration_error(
        preds, target, n_bins=15
    )


def create_config(embed_d):
    train_config = TrainConfig(
        model_config=TransformerConfig(
            embed_d=embed_d, mlp_dim=10, n_seq=2, batch_size=500
        ),
        data_config=DataSpiralsConfig(),
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam, kwargs=dict(weight_decay=0.01)
        ),
        save_nth_epoch=20,
        loss=torch.nn.BCELoss(),
        batch_size=500,
        ensemble_id=0,
    )
    train_eval = TrainEval(
        metrics=[
            lambda: Metric(
                tm.functional.accuracy,
                metric_kwargs=dict(task="binary", multidim_average="samplewise"),
            ),
            lambda: Metric(bce),
            lambda: Metric(loss),
        ],
    )
    train_run = TrainRun(
        train_config=train_config,
        train_eval=train_eval,
        epochs=500,
    )
    return train_run


if __name__ == "__main__":
    ablation(Path(__file__).parent / "results", create_config, [1, 5, 20, 50, 100])
