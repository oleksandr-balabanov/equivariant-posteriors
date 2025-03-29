import dataclasses
import json
import hashlib
import numpy as np


def json_default(data):
    try:
        return data.__name__
    except:
        return type(data).__name__


def serialize_dataclass(instance) -> dict:
    if dataclasses.is_dataclass(instance):
        return {
            "__class__": type(instance).__name__,
            "__data__": {
                k: serialize_dataclass(v) for k, v in instance.__dict__.items()
            },
        }
    elif isinstance(instance, list):
        return [serialize_dataclass(i) for i in instance]
    elif isinstance(instance, np.ndarray):
        return serialize_dataclass(instance.tolist())
    elif isinstance(instance, dict):
        return {k: serialize_dataclass(v) for k, v in instance.items()}
    else:
        return instance


def json_dumps_dataclass_str(data_class, indent=None):
    return json.dumps(
        serialize_dataclass(data_class),
        default=json_default,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        separators=(",", ":"),
    )
    
def json_dumps_dataclass(data_class):
    return json_dumps_dataclass_str(data_class).encode("utf-8")


def stable_hash(data_class):
    json_str = json_dumps_dataclass(data_class)
    return hashlib.md5(json_str).digest().hex()


def stable_hash_small(data_class):
    return stable_hash(data_class)[:7]
