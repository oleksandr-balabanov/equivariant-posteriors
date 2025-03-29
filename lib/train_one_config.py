import os
import sys

from lib.ddp import ddp_setup
from lib.train import load_or_create_state
from lib.train import do_training_unlocked

from lib.train_distributed import get_train_run_from_hash


def do_train_run(train_run_hash, device_id):
    print(f"[Single config train] Device: {device_id}")
    print(f"[Single config train] Train run: {train_run_hash}")
    train_run = get_train_run_from_hash(train_run_hash)
    state = load_or_create_state(train_run, device_id)
    do_training_unlocked(train_run, state, device_id)


if __name__ == "__main__":
    if os.getenv("EP_UNLOCKED", None) is None:
        print("This should only be called from a locked context.")
        exit(1)
    device = ddp_setup()
    do_train_run(sys.argv[1], device)
