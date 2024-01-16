#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List
from experiments.lora_ensembles.train.lora_ens_train_config_dataclass import LoraEnsTrainConfig
from experiments.lora_ensembles.pretrained_models.pretrained_models_checkpoints import (
    LLaMA_CHECKPOINT,
    MISTRAL_CHECKPOINT
)

# Lora Ens Eval Config DataClass
@dataclass
class LoraEnsEvalConfig:
    n_members: int = 1
    min_train_epochs: int = 2
    max_train_epochs: int = 4
    max_len_eval: int = 128
    eval_dataset_1_config: object = None
    eval_dataset_2_config: object = None
    eval_dir_name: str = "ens_llm_lora_evaluate"
    load_softmax_probs: bool = True

    lora_ens_train_config: LoraEnsTrainConfig = field(default_factory=lambda: LoraEnsTrainConfig(
        epochs=0,
        checkpoint=MISTRAL_CHECKPOINT,
        batch_size=8,
        learning_rate=0.000005,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0,
        lora_l2=0,
        regular_l2=0.01,
        target_modules=["q_proj", "v_proj"],
        max_len_train=128,
        max_len_val=128,
    ))