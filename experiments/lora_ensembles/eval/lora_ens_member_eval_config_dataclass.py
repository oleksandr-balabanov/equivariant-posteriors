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
    eval_dir_name: str = "/eval_data/" # directory to which the eval data is saved to
    eval_tokens:List[str] = field(default_factory=lambda: MISTRAL_EVAL_QA_TOKENS)# None # List of tokens to project to the outputs. For QA: field(default_factory=lambda: MISTRAL_EVAL_QA_TOKENS)
    eval_dataset: str = "commonsense_qa"
    eval_batch_size: int = 2
    eval_metric_type: str = "single_token"  # Can be either "single_token" or "next_token"
    max_len_eval: int = 512

    # for loading the trained checkpoint
    lora_ens_train_config: LoraEnsTrainConfig = field(default_factory=lambda: LoraEnsTrainConfig(
        checkpoint=MISTRAL_CHECKPOINT,
        train_dataset="commonsense_qa",
        epochs=1,
        batch_size=8,
        learning_rate=5e-05,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_l2=1.0,
        regular_l2=0.0,
        max_len_train = 128,
        max_len_val = 128,
        use_generative_next_token_loss = True,
        target_modules = ["q_proj", "v_proj"])
    )
