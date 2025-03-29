#!/usr/bin/env python
import torch
from pathlib import Path
import pandas as pd
import tqdm

import torchmetrics as tm

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.train_dataclasses import TrainEpochSpec
from lib.metric import MetricSample, Metric

from lib.classification_metrics import create_classification_metrics

# from lib.data_factory import DataCIFAR2Config
from lib.data_factory import DataCIFARConfig

from lib.datasets.cifar_visualization import visualize_cifar

import lib.data_factory as data_factory
from lib.models.resnet import ResnetConfig
from lib.models.mlp_proj import MLPProjClassConfig

# from lib.models.conv_small import ConvSmallConfig
from lib.lyapunov import lambda1
from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.ensemble import symlink_checkpoint_files
from lib.uncertainty import uncertainty
from lib.files import prepare_results
from lib.train import load_or_create_state, validate

import rplot


def create_config(ensemble_id):
    loss = torch.nn.CrossEntropyLoss()

    def ce_loss(outputs, targets):
        return loss(outputs["logits"], targets)

    train_config = TrainConfig(
        # model_config=MLPClassConfig(widths=[50, 50]),
        model_config=ResnetConfig(num_blocks=[3, 3, 3]),
        train_data_config=DataCIFARConfig(),
        val_data_config=DataCIFARConfig(validation=True),
        loss=ce_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.SGD,
            kwargs=dict(weight_decay=5e-4, lr=0.005, momentum=0.9, nesterov=True),
            # kwargs=dict(weight_decay=0.0, lr=0.001),
        ),
        batch_size=512,
        ensemble_id=ensemble_id,
    )
    train_eval = create_classification_metrics(visualize_cifar, 10)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=16),
        train_config=train_config,
        train_eval=train_eval,
        epochs=200,
        save_nth_epoch=1,
        validate_nth_epoch=5,
    )
    return train_run


def load_model(model: torch.nn.Module, train_run: TrainRun):
    state = torch.load("model.pt")
    model.model.load_state_dict(state, strict=False)
    model.model.eval()
    print("Loaded model.pt")
    return model


def freeze(model: torch.nn.Module, train_run: TrainRun):
    for param in model.model.parameters():
        param.requires_grad = False

    model.model.eval()
    # for layer in model.mlps[:-1]:
    # for param in layer.parameters():
    # param.requires_grad = False
    return model


def create_config_proj(ensemble_id):
    conv_config = create_config(0).train_config.model_config
    loss = torch.nn.CrossEntropyLoss()

    def cross_entropy(outputs, targets):
        # cluster_loss = 0
        # repulsive_loss = 0
        # centroids = []
        # for cidx in range(10):
        #     idxs = targets == cidx
        #     centroid = outputs["projection"][idxs].mean(dim=0)
        #     centroids.append(centroid)
        #     cluster_loss += ((outputs["projection"][idxs] - centroid)**2).sum()

        # # centroids = torch.concat(centroids, 0)

        # for cidx in range(10):
        #     idxs = targets == cidx
        #     ridxs = list(range(10))
        #     ridxs.remove(cidx)
        #     for ridx in ridxs:
        #         r2 = ((outputs["projection"][idxs] - centroids[ridx])**2).sum(dim=-1)
        #         repulsive_loss += (1.0 / (r2 + 0.01)).sum()

        # breakpoint()

        return loss(outputs["logits"], targets)  # + cluster_loss + repulsive_loss
        # return cluster_loss + repulsive_loss

    train_config = TrainConfig(
        model_config=MLPProjClassConfig(conv_config, n_proj=2),
        post_model_create_hook=load_model,
        model_pre_train_hook=freeze,
        train_data_config=DataCIFARConfig(),
        val_data_config=DataCIFARConfig(validation=True),
        loss=cross_entropy,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam,
            kwargs=dict(lr=0.001, weight_decay=1e-4)
            # kwargs=dict(weight_decay=1e-4, lr=0.001, momentum=0.9),
            # kwargs=dict(weight_decay=0e-4, lr=0.005, momentum=0.9, nesterov=True),
        ),
        batch_size=512,
        ensemble_id=ensemble_id,
        _version=2,
    )
    train_eval = create_classification_metrics(None, 10)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=16),
        train_config=train_config,
        train_eval=train_eval,
        epochs=100,
        save_nth_epoch=1,
        validate_nth_epoch=5,
    )
    return train_run


if __name__ == "__main__":
    device_id = ddp_setup()

    ensemble_config = create_ensemble_config(create_config, 5)
    ensemble = create_ensemble(ensemble_config, device_id)

    # state = load_or_create_state(ensemble_config.members[0], device_id)

    # train_epoch_spec = TrainEpochSpec(
    #     loss=ensemble_config.members[0].train_config.loss,
    #     device_id=device_id,
    # )
    # validate(state, train_epoch_spec, ensemble_config.members[0])
    # breakpoint()

    result_path = prepare_results(
        Path(__file__).parent, Path(__file__).stem, ensemble_config
    )
    torch.save(ensemble.members[0].state_dict(), "model.pt")

    ensemble_config_proj = create_ensemble_config(create_config_proj, 1)
    ensemble_proj = create_ensemble(ensemble_config_proj, device_id)
    symlink_checkpoint_files(ensemble, result_path)
    symlink_checkpoint_files(ensemble_proj, result_path)

    # diff_exists = False
    # for key in ensemble.members[0].state_dict():
    #     diff = (ensemble.members[0].state_dict()[key] - ensemble_proj.members[0].model.state_dict()[key])
    #     if diff.sum() > 0.000001:
    #         print(f"{key}: {diff.sum()}")
    #         diff_exists = True

    # if diff_exists:
    #     breakpoint()

    # print(ensemble.members[0].bn1.running_mean)
    # print(ensemble_proj.members[0].model.bn1.running_mean)
    # breakpoint()

    dsu = data_factory.get_factory().create(DataCIFARConfig(validation=True))
    dataloaderu = torch.utils.data.DataLoader(
        dsu,
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )

    uq = uncertainty(dataloaderu, ensemble, device_id)

    # fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    lambdas = []
    projections = []

    def just_logits(x):
        return ensemble.members[0](x)["logits"]

    acc_proj = Metric(
        tm.functional.accuracy,
        metric_kwargs=dict(task="multiclass", num_classes=10),
    )
    acc = Metric(
        tm.functional.accuracy,
        metric_kwargs=dict(task="multiclass", num_classes=10),
    )
    # print(ensemble.members[0].bn1.running_mean)
    # print(ensemble_proj.members[0].model.bn1.running_mean)
    # breakpoint()
    for xs, ys, ids in tqdm.tqdm(dataloaderu):
        xs = xs.to(device_id)
        ys = ys.to(device_id)

        # breakpoint()
        output = ensemble_proj.members[0](xs)
        output2 = ensemble.members[0](xs)

        metric_sample_proj = MetricSample(
            output=output["base_model_logits"],
            prediction=output["base_model_predictions"],
            target=ys,
            sample_id=ids,
            epoch=0,
        )
        metric_sample = MetricSample(
            output=output2["logits"],
            prediction=output2["predictions"],
            target=ys,
            sample_id=ids,
            epoch=0,
        )
        acc_proj(metric_sample_proj)
        acc(metric_sample)
        # if acc_proj.mean(0) != acc.mean(0):
        #     for key in ensemble.members[0].state_dict():
        #         diff = (ensemble.members[0].state_dict()[key] - ensemble_proj.members[0].model.state_dict()[key])
        #         if diff.sum() > 0.0001:
        #             print(f"{key}: {diff.sum()}")
        #             breakpoint()
        # lambda1s = lambda1(just_logits, xs.reshape(xs.shape[0], -1)) / len(
        #     ensemble_config.members[0].train_config.model_config.widths
        # )
        # lambda1s = lambda1(just_logits, xs.reshape(xs.shape[0], -1)) / 5.0
        lambda1s = lambda1(just_logits, xs) / 20.0
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
    # fig.tight_layout()
    # plt.show()
    # plt.savefig(Path(__file__).parent / "uq_lambda_mnist.pdf")
