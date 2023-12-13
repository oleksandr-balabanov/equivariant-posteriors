import psycopg
import pandas
import json
from threading import Thread
import os
import queue
from typing import Union
from pathlib import Path
from itertools import islice

from lib.train_dataclasses import TrainEpochState
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import EnsembleConfig

from lib.compute_env import env


# TODO explain analyze, psql \x, web explain analyze, CTE,
# TODO one table per value type


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def dict_to_normalized_json(input_dict):
    return json.loads(pandas.json_normalize(input_dict).to_json(orient="records"))[0]


def setup_psql():
    hostname = env().postgres_host  # "localhost"
    port = env().postgres_port  # 5432
    if "EP_POSTGRES" in os.environ:
        hostname = os.environ.get("EP_POSTGRES")
        # print(f"PSQL: {hostname}")
    if "EP_POSTGRES_PORT" in os.environ:
        try:
            port = int(os.environ.get("EP_POSTGRES_PORT"))
        except ValueError:
            pass
    with psycopg.connect(
        "dbname=postgres user=postgres password=postgres",
        host=hostname,
        port=port,
        autocommit=True,
    ) as conn:
        try:
            conn.execute(
                """
                        CREATE DATABASE equiv;
                    """
            )
        except psycopg.errors.DuplicateDatabase:
            pass

    with psycopg.connect(
        "dbname=equiv user=postgres password=postgres",
        host=hostname,
        port=port,
        autocommit=True,
    ) as conn:
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS metrics (
                    id serial PRIMARY KEY,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
                    train_id text,
                    ensemble_id text,
                    x float,
                    xaxis text,
                    variable text,
                    value float,
                    value_text text,
                    UNIQUE(train_id, x, xaxis, variable)
                )
                """
        )
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS runs (
                    id serial PRIMARY KEY,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
                    train_id text,
                    ensemble_id text,
                    variable text,
                    value_float float,
                    value_int integer,
                    value_text text,
                    UNIQUE(train_id, variable)
                )
                """
        )
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id serial PRIMARY KEY,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
                    train_id text,
                    ensemble_id text,
                    name text,
                    path text,
                    UNIQUE(train_id, name)
                )
                """
        )


def insert_param(conn, train_id, ensemble_id, variable, value):
    value_type = type(value)
    value_dispatch = {
        str: lambda v: dict(value_text=v, value_float=None, value_int=None),
        float: lambda v: dict(value_text=str(v), value_float=v, value_int=None),
        int: lambda v: dict(value_text=str(v), value_float=None, value_int=v),
        list: lambda v: dict(value_text=str(v), value_float=None, value_int=None),
        bool: lambda v: dict(value_text=str(v), value_float=None, value_int=None),
        type(None): lambda v: dict(value_text="None", value_float=None, value_int=None),
    }
    value_dict = dict(
        train_id=train_id,
        ensemble_id=ensemble_id,
        variable=variable,
    )
    if value_type not in value_dispatch:
        value_type = str
        value = str(value)
    value_dict.update(value_dispatch[value_type](value))
    conn.execute(
        """
        INSERT INTO runs (train_id, ensemble_id, variable, value_text, value_int, value_float)
        VALUES (%(train_id)s, %(ensemble_id)s, %(variable)s, %(value_text)s, %(value_int)s, %(value_float)s)
        ON CONFLICT (train_id, variable) DO UPDATE SET value_text=EXCLUDED.value_text, value_int=EXCLUDED.value_int, value_float=EXCLUDED.value_float
        """,
        value_dict,
    )


def insert_or_update_train_run(conn, train_run: TrainRun):
    train_run_flat = dict_to_normalized_json(train_run.serialize_human())
    for key, value in train_run_flat.items():
        insert_param(
            conn, train_run_flat["train_id"], train_run_flat["ensemble_id"], key, value
        )


_threads = []
_result_queue = queue.Queue()


def render_psql(train_run: TrainRun, train_epoch_state: TrainEpochState, block=False):
    # _render_psql(train_run, train_epoch_state)
    # return
    if len(_threads) > 3:
        thread = _threads.pop(0)
        train_epoch_state.timing_metric.start("psql_thread_join")
        thread.join()
        train_epoch_state.timing_metric.stop("psql_thread_join")

    thread = Thread(
        target=_render_psql,
        kwargs=dict(
            train_run=train_run,
            train_epoch_state=train_epoch_state,
            result_queue=_result_queue,
        ),
    )
    thread.start()
    _threads.append(thread)
    if block:
        thread.join()
    try:
        return _result_queue.get(block=False)
    except queue.Empty:
        return None


def get_url():
    hostname = os.getenv("EP_POSTGRES", env().postgres_host)
    port = int(os.getenv("EP_POSTGRES_PORT", env().postgres_port))
    return f"postgresql://postgres:postgres@{hostname}:{port}/equiv"


def _render_psql(
    train_run: TrainRun, train_epoch_state: TrainEpochState, result_queue: queue.Queue
):
    try:
        result_queue.put(_render_psql_unchecked(train_run, train_epoch_state))
    except psycopg.errors.OperationalError:
        result_queue.put(False)


def _render_psql_unchecked(train_run: TrainRun, train_epoch_state: TrainEpochState):
    train_run_dict = train_run.serialize_human()

    try:
        setup_psql()
    except psycopg.errors.OperationalError as e:
        # print("Couldn't setup PSQL datasink.")
        # print(str(e))
        return (False, str(e))

    train_epoch_state.timing_metric.start("psql_connection")
    with psycopg.connect(
        "dbname=equiv user=postgres password=postgres",
        host=os.getenv("EP_POSTGRES", env().postgres_host),
        port=int(os.getenv("EP_POSTGRES_PORT", env().postgres_port)),
        autocommit=False,
        prepare_threshold=None,
    ) as conn:
        train_epoch_state.timing_metric.stop("psql_connection")
        train_epoch_state.timing_metric.start("psql_queries")
        # create_param_view(conn, train_run)

        # start_time = time.time()
        train_epoch_state.timing_metric.start("psql_queries_params")
        insert_or_update_train_run(conn, train_run)
        train_epoch_state.timing_metric.stop("psql_queries_params")
        train_epoch_state.timing_metric.start("psql_queries_epoch")
        with conn.pipeline():
            for epoch in range(train_epoch_state.epoch):
                for metric in train_epoch_state.train_metrics:
                    conn.execute(
                        """
                        INSERT INTO metrics (train_id, x, xaxis, variable, value)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (train_id, x, xaxis, variable) DO UPDATE SET value = EXCLUDED.value
                    """,
                        (
                            train_run_dict["train_id"],
                            epoch,
                            "epoch",
                            metric.name(),
                            metric.mean(epoch),
                        ),
                    )
                for metric in train_epoch_state.validation_metrics:
                    if metric.mean(epoch) is not None:
                        conn.execute(
                            """
                            INSERT INTO metrics (train_id, x, xaxis, variable, value)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (train_id, x, xaxis, variable) DO UPDATE SET value = EXCLUDED.value
                        """,
                            (
                                train_run_dict["train_id"],
                                epoch,
                                "epoch",
                                f"val_{metric.name()}",
                                metric.mean(epoch),
                            ),
                        )
        train_epoch_state.timing_metric.stop("psql_queries_epoch")
        train_epoch_state.timing_metric.start("psql_queries_batch")
        with conn.pipeline(), conn.cursor() as cur:
            for metric in train_epoch_state.train_metrics:
                for idx, value in enumerate(metric.mean_batches()):
                    group_data = (
                        train_run_dict["train_id"],
                        idx,
                        "batch",
                        f"{metric.name()}_batch",
                        value,
                    )
                    query_hash = hash(group_data)
                    if query_hash not in train_epoch_state.psql_query_cache:
                        cur.execute(
                            """
                            INSERT INTO metrics (train_id, x, xaxis, variable, value)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (train_id, x, xaxis, variable) DO UPDATE SET value = EXCLUDED.value
                        """,
                            group_data,
                        )
                        train_epoch_state.psql_query_cache.add(query_hash)

        train_epoch_state.timing_metric.stop("psql_queries_batch")
        train_epoch_state.timing_metric.start("psql_queries_timing")
        with conn.pipeline(), conn.cursor() as cur:
            for (
                timing_name,
                timing_data,
            ) in train_epoch_state.timing_metric.data.items():
                for idx, value in enumerate(timing_data):
                    group_data = (
                        train_run_dict["train_id"],
                        idx,
                        "batch",
                        f"timing_{timing_name}",
                        value,
                    )
                    query_hash = hash(group_data)
                    if query_hash not in train_epoch_state.psql_query_cache:
                        cur.execute(
                            """
                            INSERT INTO metrics (train_id, x, xaxis, variable, value)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (train_id, x, xaxis, variable) DO UPDATE SET value = EXCLUDED.value
                        """,
                            group_data,
                        )
                        train_epoch_state.psql_query_cache.add(query_hash)
        train_epoch_state.timing_metric.stop("psql_queries_timing")
        train_epoch_state.timing_metric.stop("psql_queries")
        train_epoch_state.timing_metric.start("psql_commit")
        train_epoch_state.timing_metric.start("psql_commit_with_conn")
        conn.commit()
        train_epoch_state.timing_metric.stop("psql_commit")

    train_epoch_state.timing_metric.stop("psql_commit_with_conn")

    return (True, "")
    # print(f"Updated psql {time.time() - start_time}s")


def add_artifact(train_run: TrainRun, name: str, path: Union[str, Path]):
    # train_run_dict = train_run.serialize_human()
    try:
        setup_psql()
    except psycopg.errors.OperationalError as e:
        print("[Database] Could not connect to database, artifact not added.")
        return (False, str(e))

    train_run_dict = train_run.serialize_human()

    if isinstance(path, Path):
        path = path.relative_to(env().paths.artifacts)
        path = path.as_posix()

    value_dict = dict(
        train_id=train_run_dict["train_id"],
        ensemble_id=train_run_dict["ensemble_id"],
        name=name,
        path=path,
    )

    with psycopg.connect(
        "dbname=equiv user=postgres password=postgres",
        host=os.getenv("EP_POSTGRES", env().postgres_host),
        port=int(os.getenv("EP_POSTGRES_PORT", env().postgres_port)),
        autocommit=False,
    ) as conn:
        conn.execute(
            """
            INSERT INTO artifacts (train_id, ensemble_id, name, path)
            VALUES (%(train_id)s, %(ensemble_id)s, %(name)s, %(path)s)
            ON CONFLICT (train_id, name) DO UPDATE SET path=EXCLUDED.path
            """,
            value_dict,
        )
    print(f"[Database] Added artifact {name}: {path}")


def add_ensemble_artifact(
    ensemble_config: EnsembleConfig, name: str, path: Union[str, Path]
):
    add_artifact(ensemble_config.members[0], name, path)
