import os
from typing import Optional


def get_task_id() -> Optional[int]:
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        try:
            return int(os.environ.get("SLURM_ARRAY_TASK_ID"))
        except ValueError:
            return None
    else:
        return None
        # raise Exception("SLURM_ARRAY_TASK_ID not set!")
