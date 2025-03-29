import torch
import torchmetrics as tm

from lib.metric import Metric
from lib.train_dataclasses import TrainEval


def loss(output, batch):
    return torch.nn.functional.cross_entropy(
        output["predictions"], batch["target"], reduction="none"
    )


def calibration_error(output, batch):
    num_classes = output["logits"].shape[-1]
    return tm.functional.classification.calibration_error(
        output["predictions"],
        batch["target"],
        n_bins=15,
        num_classes=num_classes,
        task="multiclass",
    )


def accuracy(output, batch):
    num_classes = output["logits"].shape[-1]
    return tm.functional.accuracy(
        output["predictions"],
        batch["target"],
        task="multiclass",
        num_classes=num_classes,
    )


def create_classification_metric_dict(n_classes):
    return dict(
        accuracy=Metric(
            tm.functional.accuracy,
            metric_kwargs=dict(task="multiclass", num_classes=n_classes),
        ),
        loss=Metric(loss, raw_output=True),
        calibration=Metric(calibration_error, metric_kwargs=dict(n_classes=n_classes)),
    )


def create_classification_metric_list(n_classes):
    return [
        lambda: Metric(accuracy),
        lambda: Metric(loss),
        lambda: Metric(calibration_error),
    ]


def create_classification_metrics(data_visualizer, n_classes):
    return TrainEval(
        train_metrics=create_classification_metric_list(n_classes),
        validation_metrics=create_classification_metric_list(n_classes),
        data_visualizer=data_visualizer,
    )
