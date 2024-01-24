#!/usr/bin/env python
from lib.distributed_trainer import distributed_train
from lib.generic_ablation import generic_ablation_configs
from lib.train_distributed import request_train_runs

from experiments.lora_ensembles.mmlu_configs import create_mmlu_config


def main():
    print("Start")

    configs = generic_ablation_configs(
        create_mmlu_config,
        dict(
            ensemble_id=list(range(5)),
            num_gpus=[1],
            dataset=["mmlu_stem", "mmlu_ss"],
        ),
    )
    request_train_runs(configs)
    distributed_train(configs)
    print("ensemble finished")


if __name__ == "__main__":
    main()
