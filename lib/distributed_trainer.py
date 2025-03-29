import os
import filelock
import time
from typing import List
from filelock import FileLock
from subprocess import Popen

from lib.ddp import ddp_setup
from lib.train_dataclasses import TrainRun
from lib.train import load_or_create_state
from lib.train import do_training
from lib.stable_hash import stable_hash_small
from lib.paths import get_checkpoint_path, get_lock_path

from lib.train_distributed import fetch_requested_train_run, report_done
from lib.serialization import (
    get_serialization_epoch,
    DeserializeConfig,
)


def do_train_run(distributed_train_run, device_id):
    checkpoint_path = get_lock_path(distributed_train_run.train_run.train_config)
    print(checkpoint_path)
    if distributed_train_run.train_run.compute_config.distributed:
        lock = FileLock(f"{checkpoint_path}", 1)
        with lock:
            env = os.environ.copy()
            env["TOKENIZERS_PARALLELISM"] = "false"
            env["EP_TORCHRUN"] = "1"
            env["EP_UNLOCKED"] = "1"
            env[
                "EP_NUM_GPUS"
            ] = f"{distributed_train_run.train_run.compute_config.num_gpus}"
            command = [
                "torchrun",
                "--nnodes",
                "1",
                "--nproc_per_node",
                f"{distributed_train_run.train_run.compute_config.num_gpus}",
                "--rdzv_backend",
                "c10d",
                "--rdzv_endpoint",
                "localhost:0",
                "-m",
                "lib.train_one_config",
                distributed_train_run.hash,
            ]
            process = Popen(command, env=env)
            process.wait()
            print("Subprocess done.")
    else:
        state = load_or_create_state(distributed_train_run.train_run, device_id)
        do_training(distributed_train_run.train_run, state, device_id)


def distributed_train(requested_configs: List[TrainRun] = None):
    device_id = ddp_setup()
    print(f"[Device] {device_id}")
    last_aquired_training = time.time()
    while True:
        print("Trying to fetch train run...")
        distributed_train_run = fetch_requested_train_run(requested_configs)
        if distributed_train_run is None and requested_configs is not None:
            break
        if distributed_train_run is not None:
            try:
                last_aquired_training = time.time()
                serialized_epoch = get_serialization_epoch(
                    DeserializeConfig(
                        train_run=distributed_train_run.train_run,
                        device_id=device_id,
                    )
                )
                if (
                    serialized_epoch is None
                    or serialized_epoch < distributed_train_run.train_run.epochs
                ):
                    do_train_run(distributed_train_run, device_id)
                else:
                    print("[Distributed train] Already done, moving on...")
                report_done(distributed_train_run)
            except filelock.Timeout:
                print(
                    f"Couldn't aquire training lock for {stable_hash_small(distributed_train_run.train_run.train_config)}"
                )
            finally:
                distributed_train_run.lock.release()
        if time.time() > last_aquired_training + 10 * 60:
            print("10 minutes since last aquired training, stopping...")
            break
        time.sleep(1)


if __name__ == "__main__":
    distributed_train()
