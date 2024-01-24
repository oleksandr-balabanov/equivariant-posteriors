#!/usr/bin/env python
from lib.ensemble import create_ensemble_config
from lib.ensemble import request_ensemble
from lib.files import prepare_results
from lib.distributed_trainer import distributed_train
from experiments.lora_ensembles.train.lora_ens_train_config import create_lora_ens_train_run_config

def main():
    print("Start")
    ensemble_config = create_ensemble_config(create_lora_ens_train_run_config, 1)
    prepare_results("lora_ensemble", ensemble_config.members)
    print("ensemble_config finished")
    request_ensemble(ensemble_config)
    distributed_train(ensemble_config.members)
    print("ensemble finished")

if __name__ == "__main__":
    main()
