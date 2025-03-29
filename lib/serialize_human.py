import numpy as np
from dataclasses import is_dataclass


def serialize_human(instance):
    if hasattr(instance, "serialize_human"):
        return instance.serialize_human()
    elif is_dataclass(instance):
        return {
            "__class__": type(instance).__name__,
            "__data__": {k: serialize_human(v) for k, v in instance.__dict__.items()},
        }
    elif isinstance(instance, list):
        return [serialize_human(i) for i in instance]
    elif isinstance(instance, np.ndarray):
        return serialize_human(instance.tolist())
    elif isinstance(instance, dict):
        return {k: serialize_human(v) for k, v in instance.items()}
    else:
        return f"{instance}"
