import pandas
import json

from lib.train_dataclasses import TrainEpochState
from lib.train_dataclasses import TrainRun


def dict_to_normalized_json(input_dict):
    return json.loads(pandas.json_normalize(input_dict).to_json(orient="records"))[0]


def render_dataframe(train_run: TrainRun, train_epoch_state: TrainEpochState):
    config_cols = [
        value
        for key, value in dict_to_normalized_json(train_run.serialize_human()).items()
    ]
    config_headers = [
        key
        for key, value in dict_to_normalized_json(train_run.serialize_human()).items()
    ]
    train_metric_headers = [metric.name() for metric in train_epoch_state.train_metrics]
    val_metric_headers = [
        f"val_{metric.name()}" for metric in train_epoch_state.validation_metrics
    ]
    headers = ["epoch"] + config_headers + train_metric_headers + val_metric_headers
    rows = []
    for epoch in range(train_epoch_state.epoch):
        train_metric_values = [
            metric.mean(epoch) for metric in train_epoch_state.train_metrics
        ]
        val_metric_values = [
            metric.mean(epoch) for metric in train_epoch_state.validation_metrics
        ]
        rows.append([epoch] + config_cols + train_metric_values + val_metric_values)

    return pandas.DataFrame(columns=headers, data=rows)
