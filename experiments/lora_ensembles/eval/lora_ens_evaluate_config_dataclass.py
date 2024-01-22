#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List
from experiments.lora_ensembles.train.lora_ens_train_config_dataclass import LoraEnsTrainConfig
from experiments.lora_ensembles.datasets.dataset_config import create_dataset_config_factory
from experiments.lora_ensembles.pretrained_models.pretrained_models_checkpoints import (
    LLaMA_CHECKPOINT,
    MISTRAL_CHECKPOINT
)

# Lora Ens Eval Config DataClass
@dataclass
class LoraEnsEvalConfig:
    n_members: int = 5
    min_train_epochs: int = 1
    max_train_epochs: int = 6
    eval_dir_name: str = "/mimer/NOBACKUP/groups/snic2022-22-448/lora_ensembles/ens_llm_lora_evaluate"
    load_softmax_probs: bool = True

    train_dataset: str = "commonsense_qa"
    eval_dataset_1: str = "commonsense_qa"
    eval_dataset_2: str = "mmlu_stem"
    eval_batch_size_1: int = 8
    eval_batch_size_2: int = 4
    max_len_eval_1: int = 128
    max_len_eval_2: int = 512

    lora_ens_train_config: LoraEnsTrainConfig = field(default_factory=lambda: LoraEnsTrainConfig(
        epochs=0,
        checkpoint=MISTRAL_CHECKPOINT,
        train_dataset = "commonsense_qa",
        batch_size=8,
        learning_rate=0.000005,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0,
        lora_l2=0,
        regular_l2=0.001,
        target_modules=["q_proj", "v_proj"],
        max_len_train=128,
        max_len_val=128,
    ))
