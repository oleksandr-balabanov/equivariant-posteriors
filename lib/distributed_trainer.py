import filelock
import time
from typing import List

from lib.ddp import ddp_setup
from lib.train_dataclasses import TrainRun
from lib.train import load_or_create_state
from lib.train import do_training
from lib.stable_hash import stable_hash_small

from lib.train_distributed import fetch_requested_train_run, report_done


def do_train_run(train_run, device_id):
    state = load_or_create_state(train_run, device_id)
    do_training(train_run, state, device_id)


def distributed_train(requested_configs: List[TrainRun] = None):
    device_id = ddp_setup()
    last_aquired_training = time.time()
    while True:
        print("Trying to fetch train run...")
        distributed_train_run = fetch_requested_train_run(requested_configs)
        if distributed_train_run is not None:
            try:
                last_aquired_training = time.time()
                do_train_run(distributed_train_run.train_run, device_id)
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
