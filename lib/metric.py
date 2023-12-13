from dataclasses import dataclass
from collections.abc import Callable
from typing import Dict
import torch


@dataclass
class MetricSample:
    # output: torch.Tensor
    # prediction: torch.Tensor
    # target: torch.Tensor
    output: Dict[str, torch.Tensor]
    batch: Dict[str, torch.Tensor]
    # sample_id: torch.Tensor
    epoch: int
    batch: int


@dataclass(frozen=True)
class MetricSampleKey:
    batch_idx: int
    sample_id: int
    epoch: int


class Metric:
    def __init__(
        self,
        metric_fn: Callable[[torch.Tensor, torch.Tensor], object],
        metric_kwargs=None,
    ):
        self.values = dict()
        self.metric_fn = metric_fn
        self.metric_name = metric_fn.__name__
        self.metric_kwargs = metric_kwargs if metric_kwargs is not None else dict()
        self.mean_mem = dict()
        self.batch_idx = 0
        self.batch_values = []

    def __call__(self, metric_sample: MetricSample):
        # output = metric_sample.output.detach()
        # prediction = metric_sample.prediction.detach()
        # target = metric_sample.target.detach()
        sample_id = metric_sample.batch["sample_id"].detach()
        values = (
            self.metric_fn(
                output=metric_sample.output,
                batch=metric_sample.batch,
                **self.metric_kwargs
            )
            .detach()
            .cpu()
        )
        key = MetricSampleKey(
            sample_id=sample_id, epoch=metric_sample.epoch, batch_idx=self.batch_idx
        )
        self.batch_idx += 1
        if metric_sample.epoch in self.mean_mem:
            del self.mean_mem[metric_sample.epoch]
        self.values[key] = values.mean()
        self.batch_values.append(self.values[key].item())

    def mean(self, epoch=None):
        if epoch not in self.mean_mem:
            keys = self.values.keys()
            if epoch is not None:
                keys = list(filter(lambda key: key.epoch == epoch, keys))
            mean = None
            if len(keys) > 0:
                vals = torch.tensor([self.values[key] for key in keys])
                mean = torch.mean(vals).item()

            self.mean_mem[epoch] = mean
        return self.mean_mem[epoch]

    def mean_batches(self):
        return self.batch_values

    def serialize(self):
        return dict(values=self.values, batch_values=self.batch_values)

    def deserialize(self, serialized):
        self.values = serialized["values"]
        self.batch_values = serialized["batch_values"]

    def name(self):
        return self.metric_name


def create_metric(metric_fn):
    return lambda: Metric(metric_fn)
