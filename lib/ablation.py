import os
import torch
import plotext as plt

from lib.train import load_or_create_state
from lib.train import do_training
from lib.ddp import ddp_setup


def ablation(out_dir, create_config, values):
    out_dir.mkdir(parents=True, exist_ok=True)

    device_id = ddp_setup()
    print(f"Using device {device_id}")

    metric_ablation = {}
    for value in values:
        train_run = create_config(value)
        state = load_or_create_state(train_run, device_id)

        do_training(train_run, state, device_id)

        metric_ablation[value] = (state.validation_metrics, state.epoch)

    for metric_idx in range(len(train_run.train_eval.train_metrics)):
        plt.clf()
        plt.cld()
        plt.title(f"{state.metrics[metric_idx].name()}")
        for param, (metrics, epoch) in metric_ablation.items():
            epochs = list(range(epoch))
            means = [metrics[metric_idx].mean(epoch) for epoch in epochs]
            plt.plot(epochs, means, label=f"{param}")

        plt.show()
        plt.save_fig(out_dir / f"{state.validation_metrics[metric_idx].name()}.html")
        plt.save_fig(
            out_dir / f"{state.validation_metrics[metric_idx].name()}",
            keep_colors=True,
        )
