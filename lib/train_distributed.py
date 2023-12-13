from dataclasses import dataclass
import dill
from filelock import FileLock, Timeout
import os
from pathlib import Path
from typing import List

from lib.train_dataclasses import TrainRun
from lib.serialization import is_serialized
from lib.stable_hash import stable_hash


DISTRIBUTED_TRAINING_REQUEST_PATH = Path("distributed_training_requests/")


def get_distributed_training_request_path(train_run):
    config_hash = stable_hash(train_run)
    request_dir = DISTRIBUTED_TRAINING_REQUEST_PATH
    request_dir.mkdir(exist_ok=True, parents=True)
    request_path = request_dir / f"{config_hash}.dill"
    return request_path


def get_distributed_training_request_lock_path(train_run):
    config_hash = stable_hash(train_run)
    lock_dir = DISTRIBUTED_TRAINING_REQUEST_PATH
    lock_dir.mkdir(exist_ok=True, parents=True)
    lock_path = lock_dir / f"lock_{config_hash}"
    return lock_path


def get_lock_from_hash(config_hash):
    lock_dir = DISTRIBUTED_TRAINING_REQUEST_PATH
    lock_dir.mkdir(exist_ok=True, parents=True)
    lock_path = lock_dir / f"lock_{config_hash}"
    return lock_path


def get_request_path_from_hash(config_hash):
    request_dir = DISTRIBUTED_TRAINING_REQUEST_PATH
    request_dir.mkdir(exist_ok=True, parents=True)
    request_path = request_dir / f"{config_hash}.dill"
    return request_path


def get_train_run_from_hash(config_hash):
    request_path = get_request_path_from_hash(config_hash)
    with open(request_path, "rb") as request_file:
        return dill.load(request_file)


def request_train_run(train_run: TrainRun):
    with FileLock(get_distributed_training_request_lock_path(train_run), 1):
        with open(
            get_distributed_training_request_path(train_run), "wb"
        ) as request_file:
            dill.dump(train_run, request_file)
        print(
            f"Wrote request file {get_distributed_training_request_lock_path(train_run)}"
        )


@dataclass
class DistributedTrainRun:
    train_run: TrainRun
    lock: FileLock


def lock_requested_hash(hash):
    lock = FileLock(get_lock_from_hash(hash))
    try:
        lock.acquire(timeout=1)
        print(f"Aquired lock {get_lock_from_hash(hash)}")
        return lock
    except Timeout:
        print(f"Probably being trained [{hash}]")

    return None


def fetch_requested_hash(hash):
    lock = FileLock(get_lock_from_hash(hash))
    try:
        lock.acquire(timeout=1)
        print(f"Aquired lock {get_lock_from_hash(hash)}")
        if get_request_path_from_hash(hash).is_file():
            train_run = get_train_run_from_hash(hash)
            return DistributedTrainRun(train_run=train_run, lock=lock)
        else:
            print(
                f"No requested train run file present at {get_request_path_from_hash(hash)}. Releasing lock."
            )
            lock.release()
    except Timeout:
        print(f"Probably being trained [{hash}]")

    return None


def fetch_requested_train_run(train_only_from_configs: List[TrainRun] = None):
    if train_only_from_configs is None:
        train_only_from_configs = []

    hashes_to_train = [stable_hash(config) for config in train_only_from_configs]

    pool = DISTRIBUTED_TRAINING_REQUEST_PATH.glob("*.dill")
    pool_with_ctime = []
    for pool_file in pool:
        try:
            ctime = os.path.getctime(pool_file)
            pool_with_ctime.append((ctime, pool_file))
        except:
            pass
    sorted_pool = [x[1] for x in sorted(pool_with_ctime)]
    for request_file in sorted_pool:
        hash = request_file.stem
        if len(hashes_to_train) > 0 and hash not in hashes_to_train:
            continue
        lock = FileLock(get_lock_from_hash(hash))
        try:
            lock.acquire(timeout=1)
            print(f"Aquired lock {get_lock_from_hash(hash)}")
            if get_request_path_from_hash(hash).is_file():
                train_run = get_train_run_from_hash(hash)
                return DistributedTrainRun(train_run=train_run, lock=lock)
            else:
                lock.release()
        except Timeout:
            print(f"Probably being trained [{hash}]...trying another")

    return None


def report_done(distributed_train_run: DistributedTrainRun):
    get_distributed_training_request_path(distributed_train_run.train_run).unlink()
    distributed_train_run.lock.release()
    print(f"Worker reported {stable_hash(distributed_train_run.train_run)} done")
