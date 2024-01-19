#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List
from experiments.lora_ensembles.pretrained_models.pretrained_models_checkpoints import (
    LLaMA_CHECKPOINT,
    MISTRAL_CHECKPOINT
)

# Ens Train Config dataclass
@dataclass
class LoraEnsTrainConfig:
    checkpoint:str=MISTRAL_CHECKPOINT
    train_dataset:str="mmlu_stem"
    epochs:int=10
    batch_size:int=8
    learning_rate:float=0.000005
    lora_rank:int=8
    lora_alpha:int=32
    lora_dropout:float=0
    lora_l2:float=0
    regular_l2:float=0.01
    max_len_train:int = 128
    max_len_val:int = 128
    target_modules:List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

