#!/usr/bin/env python
import torch

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.classification_metrics import create_classification_metrics
from lib.data_registry import DataMNISTConfig
from lib.datasets.mnist_visualization import visualize_mnist
from lib.models.transformer import TransformerConfig
from lib.generic_ablation import generic_ablation
from lib.distributed_trainer import distributed_train


def create_config(mlp_dim, ensemble_id):
    ce_loss = torch.nn.CrossEntropyLoss()

    def loss(output, batch):
        return ce_loss(output["logits"], batch["target"])

    train_config = TrainConfig(
        # model_config=MLPClassConfig(width=mlp_dim),
        model_config=TransformerConfig(
            embed_d=32,
            mlp_dim=64,
            n_seq=784 // (14 * 14),
            batch_size=256,
            num_layers=2,
            num_heads=1,
        ),
        train_data_config=DataMNISTConfig(),
        val_data_config=DataMNISTConfig(validation=True),
        loss=loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam, kwargs=dict(lr=0.001, weight_decay=0.001)
        ),
        batch_size=256,
        ensemble_id=ensemble_id,
    )
    train_eval = create_classification_metrics(visualize_mnist, 10)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=80,
        save_nth_epoch=10,
        validate_nth_epoch=5,
    )
    return train_run


if __name__ == "__main__":
    configs = generic_ablation(
        create_config,
        dict(mlp_dim=[100, 10, 50, 200], ensemble_id=list(range(5))),
    )
    distributed_train(configs)
