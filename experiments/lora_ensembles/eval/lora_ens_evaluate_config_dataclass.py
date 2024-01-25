#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List
from experiments.lora_ensembles.train.lora_ens_train_config_dataclass import LoraEnsTrainConfig
from experiments.lora_ensembles.datasets.dataset_config import create_dataset_config_factory
from experiments.lora_ensembles.pretrained_models.pretrained_models_checkpoints import (
    LLaMA_CHECKPOINT,
    MISTRAL_CHECKPOINT
)
from experiments.lora_ensembles.eval.lora_ens_evaluate_tokens_consts import MISTRAL_EVAL_QA_TOKENS

# Lora Ens Eval Config DataClass
@dataclass
class LoraEnsEvalConfig:
    min_train_epochs: int = 1
    max_train_epochs: int = 6
    eval_dir_name: str = "/mimer/NOBACKUP/groups/snic2022-22-448/lora_ensembles/ens_llm_lora_evaluate"
    load_softmax_probs: bool = True
    eval_tokens:List[str] = field(default_factory=lambda: MISTRAL_EVAL_QA_TOKENS)
    train_dataset: str = "mmlu_ss"

    eval_dataset_1: str = "mmlu_ss"
    eval_batch_size_1: int = 4
    max_len_eval_1: int = 512

    eval_ood: bool = True
    eval_dataset_2: str = "mmlu_stem"
    eval_batch_size_2: int = 4
    max_len_eval_2: int = 512

    n_members_1: int = 5
    lora_ens_train_config_1: LoraEnsTrainConfig = field(default_factory=lambda: LoraEnsTrainConfig(
        epochs=5,
        checkpoint=MISTRAL_CHECKPOINT,
        train_dataset = "mmlu_ss",
        batch_size=2,
        learning_rate=0.000005/4,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_l2=0.1,
        regular_l2=0,
        target_modules=["q_proj", "v_proj"],
        max_len_train=512,
        max_len_val=512,
    ))

    eval_agr_var:bool = False
    n_members_2: int = 5
    lora_ens_train_config_2: LoraEnsTrainConfig = field(default_factory=lambda: LoraEnsTrainConfig(
        epochs=0,
        checkpoint=MISTRAL_CHECKPOINT,
        train_dataset = "commonsense_qa",
        batch_size=8,
        learning_rate=0.000005,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0,
        lora_l2=1,
        regular_l2=0,
        target_modules=["q_proj", "v_proj"],
        max_len_train=128,
        max_len_val=128,
    ))
