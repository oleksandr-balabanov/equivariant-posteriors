import os
from pathlib import Path
from lib.compute_env_config import ComputeEnvironment, Paths

ENTVAR = os.environ.get("ENTVAR", "/default/path") 

env = ComputeEnvironment(
    paths=Paths(
        checkpoints=Path(ENTVAR) / "lora_ensembles" / "checkpoints",
        artifacts=Path(ENTVAR) / "lora_ensembles" / "artifacts",
        datasets=Path(ENTVAR) / "lora_ensembles" / "datasets",
    )
)

