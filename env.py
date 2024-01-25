from pathlib import Path
from lib.compute_env_config import ComputeEnvironment, Paths

env = ComputeEnvironment(
    paths=Paths(
        checkpoints=Path("/mimer/NOBACKUP/groups/snic2022-22-448/lora_ensembles/checkpoints_hamlin"),
        artifacts=Path("/mimer/NOBACKUP/groups/snic2022-22-448/lora_ensembles/artifacts"),
        datasets=Path("/mimer/NOBACKUP/groups/snic2022-22-448/lora_ensembles/datasets"),
        
    ),
    postgres_host="alvis2.c3se.chalmers.se",
)

