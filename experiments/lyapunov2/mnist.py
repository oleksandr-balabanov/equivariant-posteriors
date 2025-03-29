#!/usr/bin/env python
import torch
from pathlib import Path
import pandas as pd
import tqdm

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.classification_metrics import create_classification_metrics
from lib.data_factory import DataMNISTConfig
from lib.datasets.mnist_visualization import visualize_mnist

import lib.data_factory as data_factory
from lib.models.mlp_proj import MLPProjClassConfig
from lib.models.mlp import MLPClassConfig
from lib.lyapunov import lambda1
from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.uncertainty import uncertainty
from lib.files import prepare_results

import rplot


def create_config(ensemble_id):
    loss = torch.nn.CrossEntropyLoss()

    def ce_loss(outputs, targets):
        return loss(outputs["logits"], targets)

    train_config = TrainConfig(
        # model_config=MLPClassConfig(widths=[50, 50]),
        model_config=MLPClassConfig(widths=[20] * 16),
        train_data_config=DataMNISTConfig(),
        val_data_config=DataMNISTConfig(validation=True),
        loss=ce_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.SGD,
            kwargs=dict(weight_decay=1e-4, lr=0.05, momentum=0.9),
            # kwargs=dict(weight_decay=0.0, lr=0.001),
        ),
        batch_size=2**13,
        ensemble_id=ensemble_id,
    )
    train_eval = create_classification_metrics(visualize_mnist, 10)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=16),
        train_config=train_config,
        train_eval=train_eval,
        epochs=300,
        save_nth_epoch=1,
        validate_nth_epoch=5,
    )
    return train_run


def load_model(model: torch.nn.Module, train_run: TrainRun, result_path: Path):
    state = torch.load(result_path / "model.pt")
    model.model.load_state_dict(state, strict=False)
    return model


def freeze(model: torch.nn.Module, train_run: TrainRun):
    for param in model.model.parameters():
        param.requires_grad = False
    # for layer in model.mlps[:-1]:
    # for param in layer.parameters():
    # param.requires_grad = False
    return model


def create_config_proj(ensemble_id, result_path):
    mlp_config = create_config(0).train_config.model_config
    loss = torch.nn.CrossEntropyLoss()

    def cross_entropy(outputs, targets):
        return loss(outputs["logits"], targets)

    train_config = TrainConfig(
        model_config=MLPProjClassConfig(mlp_config, 2),
        post_model_create_hook=lambda model, train_run: load_model(
            model, train_run, result_path
        ),
        model_pre_train_hook=freeze,
        train_data_config=DataMNISTConfig(),
        val_data_config=DataMNISTConfig(validation=True),
        loss=cross_entropy,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.SGD,
            kwargs=dict(weight_decay=1e-4, lr=0.05, momentum=0.9),
        ),
        batch_size=2**13,
        ensemble_id=ensemble_id,
        _version=1,
    )
    train_eval = create_classification_metrics(visualize_mnist, 10)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=16),
        train_config=train_config,
        train_eval=train_eval,
        epochs=30,
        save_nth_epoch=1,
        validate_nth_epoch=5,
    )
    return train_run


if __name__ == "__main__":
    device_id = ddp_setup()

    ensemble_config = create_ensemble_config(create_config, 3)
    ensemble = create_ensemble(ensemble_config, device_id)

    result_path = prepare_results(
        Path(__file__).parent, Path(__file__).stem, ensemble_config
    )

    torch.save(ensemble.members[0].state_dict(), result_path / "model.pt")

    ensemble_config_proj = create_ensemble_config(
        lambda ensemble_id: create_config_proj(ensemble_id, result_path), 1
    )
    ensemble_proj = create_ensemble(ensemble_config_proj, device_id)

    dsu = data_factory.get_factory().create(DataMNISTConfig(validation=True))
    dataloaderu = torch.utils.data.DataLoader(
        dsu,
        batch_size=10,
        shuffle=False,
        drop_last=False,
    )

    uq = uncertainty(dataloaderu, ensemble, device_id)

    # fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    lambdas = []
    projections = []

    def just_logits(x):
        return ensemble.members[0](x)["logits"]

    for xs, ys, ids in tqdm.tqdm(dataloaderu):
        xs = xs.to(device_id)

        output = ensemble_proj.members[0](xs)
        lambda1s = lambda1(just_logits, xs.reshape(xs.shape[0], -1)) / len(
            ensemble_config.members[0].train_config.model_config.widths
        )
        lambdas.append(lambda1s)

        projections.append(output["projection"].detach()[:, :2])
        X = output["projection"].detach()[:, 0]
        Y = output["projection"].detach()[:, 1]
        C = lambda1s

    lambda1_tensor = torch.concat(lambdas, dim=0)
    projection_tensor = torch.concat(projections, dim=0)

    data = torch.concat(
        [
            lambda1_tensor[:, None].cpu(),
            uq.MI[:, None].cpu(),
            uq.H[:, None].cpu(),
            uq.sample_ids[:, None].cpu(),
            projection_tensor.cpu(),
            uq.mean_pred[:, None].cpu(),
        ],
        dim=-1,
    )
    df = pd.DataFrame(
        columns=["lambda", "MI", "H", "id", "x", "y", "pred"], data=data.numpy()
    )
    rplot.plot_r(df, result_path)
