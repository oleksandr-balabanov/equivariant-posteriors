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
            ensemble_id=list(range(1)),
            num_gpus=[1],
            dataset=["mmlu_stem", "mmlu_ss"],
            lora_dropout=[0.0, 0.1, 0.2, 0.3],
            lora_l2=[10.0, 1.0, 0.1, 0.01],
        ),
    )
    request_train_runs(configs)
    distributed_train(configs)
    print("ensemble finished")


if __name__ == "__main__":
    main()
