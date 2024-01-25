from lib.train_dataclasses import ComputeConfig
from experiments.lora_ensembles.train.lora_ens_train_config import (
    create_lora_ens_train_run_config,
)
from experiments.lora_ensembles.train.lora_ens_train_config_dataclass import (
    LoraEnsTrainConfig,
)


def create_mmlu_config(ensemble_id, num_gpus, dataset, lora_dropout=0, lora_l2=1):
    batch_size = 2
    compute_config = ComputeConfig(num_workers=5, distributed=True, num_gpus=num_gpus)
    return create_lora_ens_train_run_config(
        ensemble_id,
        LoraEnsTrainConfig(
            train_dataset=dataset,
            learning_rate=0.000005 * batch_size * compute_config.num_gpus / 8,
            batch_size=batch_size,
            max_len_train=512,
            max_len_val=512,
            epochs=15,
            lora_l2=lora_l2,
            lora_dropout=lora_dropout,
        ),
        compute_config=compute_config,
    )
