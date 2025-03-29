import torch
import torchmetrics as tm

from lib.metric import Metric
from lib.train_dataclasses import TrainEval


def create_regression_metric_list(loss):
    def non_reducing_loss(output, batch):
        return loss(output["predictions"], batch["target"], reduction="none")

    return [
        # lambda: Metric(
        #     tm.functional.accuracy,
        #     metric_kwargs=dict(task="multiclass", num_classes=n_classes),
        # ),
        lambda: Metric(non_reducing_loss),
        # lambda: Metric(calibration_error, metric_kwargs=dict(n_classes=n_classes)),
    ]


def create_regression_metrics(loss, data_visualizer):
    return TrainEval(
        train_metrics=create_regression_metric_list(loss),
        validation_metrics=create_regression_metric_list(loss),
        # train_metrics=[
        #     lambda: Metric(
        #         tm.functional.accuracy,
        #         metric_kwargs=dict(task="multiclass", num_classes=n_classes),
        #         # metric_kwargs=dict(num_classes=n_classes),
        #     ),
        #     lambda: Metric(loss, raw_output=True),
        #     lambda: Metric(calibration_error, metric_kwargs=dict(n_classes=n_classes)),
        #     # lambda: Metric(calibration_error),
        # ],
        # validation_metrics=[
        #     lambda: Metric(
        #         tm.functional.accuracy,
        #         metric_kwargs=dict(task="multiclass", num_classes=n_classes),
        #         # metric_kwargs=dict(num_classes=n_classes),
        #     ),
        #     lambda: Metric(calibration_error, metric_kwargs=dict(n_classes=n_classes)),
        #     # lambda: Metric(calibration_error),
        #     lambda: Metric(loss, raw_output=True),
        # ],
        data_visualizer=data_visualizer,
    )
