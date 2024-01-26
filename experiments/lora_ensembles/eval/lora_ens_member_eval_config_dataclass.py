#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List
from experiments.lora_ensembles.train.lora_ens_train_config_dataclass import LoraEnsTrainConfig
from experiments.lora_ensembles.pretrained_models.pretrained_models_checkpoints import (
    LLaMA_CHECKPOINT,
    MISTRAL_CHECKPOINT
)
from experiments.lora_ensembles.eval.lora_ens_member_eval_tokens_consts import MISTRAL_EVAL_QA_TOKENS

# Lora Ens Eval Config DataClass
@dataclass
class LoraEnsMemberEvalConfig:
    epoch: int = 0
    member_id: int = 0
    eval_dir_name: str = "/mimer/NOBACKUP/groups/snic2022-22-448/lora_ensembles/ens_llm_lora_eval_2"
    eval_tokens:List[str] = field(default_factory=lambda: MISTRAL_EVAL_QA_TOKENS)
    eval_dataset: str = "mmlu_ss"
    eval_batch_size: int = 4
    max_len_eval: int = 512
    lora_ens_train_config: LoraEnsTrainConfig = field(default_factory=lambda: LoraEnsTrainConfig(
        checkpoint=MISTRAL_CHECKPOINT,
        train_dataset="mmlu_ss",
        epochs=1,
        batch_size=2,
        effective_batch_size=2,
        learning_rate=0.000005/4,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_l2=0.1,
        regular_l2=0,
        max_len_train = 512,
        max_len_val = 512,
        target_modules = ["q_proj", "v_proj"])
    )