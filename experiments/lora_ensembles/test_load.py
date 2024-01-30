#!/usr/bin/env python
from lib.distributed_trainer import distributed_train
from lib.generic_ablation import generic_ablation_configs
from lib.train_distributed import request_train_runs
from lib.ensemble import create_ensemble_config
from lib.serialization import deserialize_model, DeserializeConfig, is_serialized
from lib.ddp import ddp_setup

from experiments.lora_ensembles.train.lora_ens_train_config import (
    create_lora_ens_train_run_config,
)
from experiments.lora_ensembles.train.lora_ens_train_config_dataclass import (
    LoraEnsTrainConfig,
)


def main():
    print("Start")
    device = ddp_setup()

    def create_ensemble(lora_ens_train_config: LoraEnsTrainConfig):
        def create_config(ensemble_id):
            return create_lora_ens_train_run_config(ensemble_id, lora_ens_train_config)

        return create_config

    ensemble_stem_config = create_ensemble_config(
        create_ensemble(
            LoraEnsTrainConfig(
                train_dataset="mmlu_stem",
                learning_rate=1.25e-6,
                batch_size=2,
                max_len_train=512,
                max_len_val=512,
                epochs=15,
                lora_l2=10.0,
                lora_dropout=0.0,
                regular_l2=0,
            )
        ),
        n_members=10,
    )
    ensemble_ss_config = create_ensemble_config(
        create_ensemble(
            LoraEnsTrainConfig(
                train_dataset="mmlu_ss",
                learning_rate=1.25e-6,
                batch_size=2,
                max_len_train=512,
                max_len_val=512,
                epochs=15,
                lora_l2=10.0,
                lora_dropout=0.0,
                regular_l2=0,
            )
        ),
        n_members=10,
    )

    ensemble_qs_regular = create_ensemble_config(
        create_ensemble(
            LoraEnsTrainConfig(
                train_dataset="commonsense_qa",
                epochs=10,
                batch_size=8,
                learning_rate=5e-06,
                lora_rank=8,
                lora_alpha=32,
                lora_dropout=0,
                lora_l2=0,
                regular_l2=1,
                max_len_train=128,
                max_len_val=128,
            )
        ),
        n_members=10,
    )

    ensemble_ss_reg = create_ensemble_config(
        create_ensemble(
            LoraEnsTrainConfig(
                train_dataset="mmlu_ss",
                # learning_rate=0.000005 * batch_size / 8,
                learning_rate=1.25e-6,
                batch_size=2,
                max_len_train=512,
                max_len_val=512,
                epochs=15,
                lora_l2=0.0,
                lora_dropout=0.0,
                regular_l2=1.0,
            )
        ),
        n_members=10,
    )
    ensemble_stem_reg = create_ensemble_config(
        create_ensemble(
            LoraEnsTrainConfig(
                train_dataset="mmlu_stem",
                # learning_rate=0.000005 * batch_size / 8,
                learning_rate=1.25e-6,
                batch_size=2,
                max_len_train=512,
                max_len_val=512,
                epochs=15,
                lora_l2=0.0,
                lora_dropout=0.0,
                regular_l2=1.0,
            )
        ),
        n_members=10,
    )

    configs = (
        ensemble_ss_config.members
        + ensemble_stem_config.members
        + ensemble_qs_regular.members
        + ensemble_ss_reg.members
        + ensemble_stem_reg.members
    )
    for config in configs:
        if not is_serialized(config):
            print(f"Couldn't load {config}")


if __name__ == "__main__":
    main()
