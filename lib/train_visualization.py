import os
import plotext as plt
import shutil
import psutil

from contextlib import redirect_stdout
import io
import torch

from lib.paths import get_or_create_checkpoint_path
from lib.stable_hash import stable_hash_small


def visualize_progress_batches(state, train_run, last_postgres_result, device):
    # plt.clt()
    plt.cld()
    plt.scatter()
    # Two columns
    plt.subplots(1, 3)

    train_metric_names = [metric.name() for metric in state.train_metrics]
    # val_metric_names = [metric.name() for metric in state.validation_metrics]

    common_metrics = list(
        set(train_metric_names)
    )  # .intersection(set(val_metric_names)))
    common_metrics = sorted(common_metrics)
    n_metrics = min(4, len(common_metrics))

    train_indices = [train_metric_names.index(name) for name in common_metrics]
    # val_indices = [val_metric_names.index(name) for name in common_metrics]

    # First column (many metrics in rows)
    plt.subplot(1, 1).subplots(n_metrics, 1)

    for idx in range(n_metrics):
        train_metric = state.train_metrics[train_indices[idx]]
        # val_metric = state.validation_metrics[val_indices[idx]]

        batches = train_metric.mean_batches()
        batch_factor = 1
        if len(batches) > 1000:
            batch_factor = len(batches) // 1000
            batches = batches[::batch_factor]
        train_means = list(enumerate(batches))
        # train_means = [
        # (epoch, mean) for (epoch, mean) in train_means if mean is not None
        # ]
        # val_means = list(enumerate(val_metric.mean_batches()))
        # val_means = [(epoch, mean) for (epoch, mean) in val_means if mean is not None]
        plt.subplot(1, 1).subplot(idx + 1, 1)
        plt.title(common_metrics[idx])
        plt.xlabel("batches")
        if len(train_means) > 0:
            x = [batch * batch_factor for (batch, mean) in train_means]
            y = [mean for (batch, mean) in train_means]
            plt.plot(x, y, label=f"Train {common_metrics[idx]}")
        # if len(val_means) > 0:
        # x = [epoch for (epoch, mean) in val_means]
        # y = [mean for (epoch, mean) in val_means]
        # plt.plot(x, y, label=f"Val {common_metrics[idx]}")

    # Second column (config)
    plt.subplot(1, 2).subplots(2, 1)
    plt.subplot(1, 2).subplot(1, 1)
    if train_run.train_eval.data_visualizer is not None:
        train_run.train_eval.data_visualizer(plt, state, device)
    plt.subplot(1, 2).subplot(2, 1).subplots(1, 3)
    plt.subplot(1, 2).subplot(2, 1).subplot(1, 1)

    if state.device_memory_stats is not None:
        plot_device_memory_stats(
            plt, filter_memory_stats(state.device_memory_stats), device
        )
    plt.subplot(1, 2).subplot(2, 1).subplot(1, 2)

    plot_host_memory_stats(plt)

    plt.subplot(1, 2).subplot(2, 1).subplot(1, 3)
    status = True
    if last_postgres_result is not None:
        status, msg = last_postgres_result
        if not status:
            plt.text(msg, 0, 0, color="red")
    background = "green" if status else "red"
    color = "white" if status else "black"
    plt.text("PSQL", 0, 1, background=background, color=color)
    # plt.text(str(last_postgres_result), 0, 0)

    plt.xaxes(False, False)
    plt.yaxes(False, False)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Third column
    plt.subplot(1, 3)
    plt.title("Config")
    # tc = "\n".join(text_config(asdict(train_run)))
    tc_config = text_config(train_run.serialize_human())
    tc_header = [
        f"train_run: {stable_hash_small(train_run)}",
        f"train_config: {stable_hash_small(train_run.train_config)}",
    ]
    tc = "\n".join(tc_header + tc_config)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.text(tc, 0, 1, color="black")
    plt.xaxes(False, False)
    plt.yaxes(False, False)
    plt.xticks([])
    plt.yticks([])

    plt.show()

    checkpoint_path = get_or_create_checkpoint_path(train_run.train_config)
    f = io.StringIO()
    with redirect_stdout(f):
        plt.save_fig(checkpoint_path / "batches_tmp.html")
        plt.save_fig(checkpoint_path / "batches_term_")
        plt.save_fig(checkpoint_path / "batches_term_color_", keep_colors=True)
    shutil.move(
        checkpoint_path / "batches_tmp.html", checkpoint_path / "batches_training.html"
    )
    shutil.move(
        checkpoint_path / "batches_term_", checkpoint_path / "batches_training.term"
    )
    shutil.move(
        checkpoint_path / "batches_term_color_",
        checkpoint_path / "batches_training.term_color",
    )


def visualize_progress(state, train_run, last_postgres_result, device):
    # plt.clt()
    plt.cld()
    plt.scatter()
    epochs = list(range(state.epoch))
    # Two columns
    plt.subplots(1, 3)

    train_metric_names = [metric.name() for metric in state.train_metrics]
    val_metric_names = [metric.name() for metric in state.validation_metrics]

    common_metrics = list(set(train_metric_names).intersection(set(val_metric_names)))
    common_metrics = sorted(common_metrics)
    n_metrics = min(4, len(common_metrics))

    train_indices = [train_metric_names.index(name) for name in common_metrics]
    val_indices = [val_metric_names.index(name) for name in common_metrics]

    # First column (many metrics in rows)
    plt.subplot(1, 1).subplots(n_metrics, 1)

    for idx in range(n_metrics):
        train_metric = state.train_metrics[train_indices[idx]]
        val_metric = state.validation_metrics[val_indices[idx]]

        train_means = [(epoch, train_metric.mean(epoch)) for epoch in epochs]
        train_means = [
            (epoch, mean) for (epoch, mean) in train_means if mean is not None
        ]
        val_means = [(epoch, val_metric.mean(epoch)) for epoch in epochs]
        val_means = [(epoch, mean) for (epoch, mean) in val_means if mean is not None]
        plt.subplot(1, 1).subplot(idx + 1, 1)
        plt.title(common_metrics[idx])
        plt.xlabel("epoch")
        if len(train_means) > 0:
            x = [epoch for (epoch, mean) in train_means]
            y = [mean for (epoch, mean) in train_means]
            plt.plot(x, y, label=f"Train {common_metrics[idx]}")
        if len(val_means) > 0:
            x = [epoch for (epoch, mean) in val_means]
            y = [mean for (epoch, mean) in val_means]
            plt.plot(x, y, label=f"Val {common_metrics[idx]}")

    # Second column (config)
    plt.subplot(1, 2).subplots(2, 1)
    plt.subplot(1, 2).subplot(1, 1)
    if train_run.train_eval.data_visualizer is not None:
        train_run.train_eval.data_visualizer(plt, state, device)
    plt.subplot(1, 2).subplot(2, 1).subplots(1, 3)
    plt.subplot(1, 2).subplot(2, 1).subplot(1, 1)

    if state.device_memory_stats is not None:
        plot_device_memory_stats(
            plt, filter_memory_stats(state.device_memory_stats), device
        )
    plt.subplot(1, 2).subplot(2, 1).subplot(1, 2)

    plot_host_memory_stats(plt)

    plt.subplot(1, 2).subplot(2, 1).subplot(1, 3)
    status = True
    if last_postgres_result is not None:
        status, msg = last_postgres_result
        if not status:
            plt.text(msg, 0, 0, color="red")
    background = "green" if status else "red"
    color = "white" if status else "black"
    plt.text("PSQL", 0, 1, background=background, color=color)
    # plt.text(str(last_postgres_result), 0, 0)

    plt.xaxes(False, False)
    plt.yaxes(False, False)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Third column
    plt.subplot(1, 3)
    plt.title("Config")
    # tc = "\n".join(text_config(asdict(train_run)))
    tc_config = text_config(train_run.serialize_human())
    tc_header = [
        f"train_run: {stable_hash_small(train_run)}",
        f"train_config: {stable_hash_small(train_run.train_config)}",
    ]
    tc = "\n".join(tc_header + tc_config)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.text(tc, 0, 1, color="black")
    plt.xaxes(False, False)
    plt.yaxes(False, False)
    plt.xticks([])
    plt.yticks([])

    plt.show()

    checkpoint_path = get_or_create_checkpoint_path(train_run.train_config)
    f = io.StringIO()
    with redirect_stdout(f):
        plt.save_fig(checkpoint_path / "tmp.html")
        plt.save_fig(checkpoint_path / "term_")
        plt.save_fig(checkpoint_path / "term_color_", keep_colors=True)
    shutil.move(checkpoint_path / "tmp.html", checkpoint_path / "training.html")
    shutil.move(checkpoint_path / "term_", checkpoint_path / "training.term")
    shutil.move(
        checkpoint_path / "term_color_", checkpoint_path / "training.term_color"
    )


def text_config(config, level=0, y=0):
    text = []
    for key, value in config.items():
        if isinstance(value, dict):
            text.append(f"{'  '*level}{key}:")
            text = text + text_config(value, level + 1)
        else:
            text.append(f"{'  '*level}{key}: {value}")
    return text


def filter_memory_stats(memory_stats: dict):
    return {
        key: value["all"]
        for key, value in memory_stats.items()
        if isinstance(value, dict)
        and "all" in value
        and key in ["allocated_bytes", "reserved_bytes", "active_bytes"]
    }


def plot_device_memory_stats(plt, memory_stats: dict, device):
    if device == "cpu":
        return
    device_stats = torch.cuda.get_device_properties(device)

    def bytes_to_mb(bytes):
        return bytes / 1e6

    # keys = list(memory_stats.keys())
    keys = ["allocated_bytes"]  # list(memory_stats.keys())
    current = [bytes_to_mb(memory_stats[key]["current"]) for key in keys]
    peak = [bytes_to_mb(memory_stats[key]["peak"]) for key in keys]
    max = [bytes_to_mb(device_stats.total_memory) for key in keys]
    num_gpus = int(os.getenv("EP_NUM_GPUS", "1"))
    plt.text(f"Device ({num_gpus} gpus)", 0, 0)
    plt.multiple_bar(
        keys, [current, peak, max], label=["current", "peak", "max"], orientation="v"
    )


def plot_host_memory_stats(
    plt,
):
    def bytes_to_mb(bytes):
        return int(bytes / 1e6)

    max = [bytes_to_mb(psutil.virtual_memory().total)]
    current = [bytes_to_mb(psutil.virtual_memory().used)]
    peak = [0]
    plt.multiple_bar(
        ["RAM"], [current, peak, max], label=["current", "peak", "max"], orientation="v"
    )
    plt.title("Host")
