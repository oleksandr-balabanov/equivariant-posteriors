#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List
import torch

from experiments.lora_ensembles.utils.generative_llm_losses import (
    generative_next_token_and_lora_l2,
    generative_next_token_loss,
    generative_single_token_and_lora_l2,
    generative_single_token_loss,
)

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import TrainEval
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig


from lib.models.llama2_generative import LLaMA2GenerativeConfig
from lib.models.mistral_generative import MistralGenerativeConfig

from lib.metric import create_metric
from lib.data_registry import DataCommonsenseQaConfig

from experiments.lora_ensembles.utils.lora_ens_metrics import accuracy, calibration_error
from experiments.lora_ensembles.train.lora_ens_train_config_dataclass import LoraEnsTrainConfig




# Configuration for Training
def create_lora_ens_train_run_config(
    ensemble_id:int,
    ens_train_config:LoraEnsTrainConfig = LoraEnsTrainConfig()
):
    train_config = TrainConfig(
        model_config=MistralGenerativeConfig(
            checkpoint=ens_train_config.checkpoint,
            lora_rank=ens_train_config.lora_rank,
            lora_alpha=ens_train_config.lora_alpha,
            lora_dropout=ens_train_config.lora_dropout,
            lora_l2=ens_train_config.lora_l2,
            target_modules=ens_train_config.target_modules,
        ),
        train_data_config=DataCommonsenseQaConfig(
            dataset="commonsense_qa",
            model_checkpoint=ens_train_config.checkpoint,
            max_len=ens_train_config.max_len_train,
            dataset_split="train",
        ),
        val_data_config=DataCommonsenseQaConfig(
            dataset="commonsense_qa",
            model_checkpoint=ens_train_config.checkpoint,
            max_len=ens_train_config.max_len_val,
            dataset_split="validation",
        ),
        loss=generative_single_token_and_lora_l2,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(
                weight_decay=ens_train_config.regular_l2, 
                lr=ens_train_config.learning_rate
            ),
        ),
        batch_size=ens_train_config.batch_size,
        ensemble_id=ensemble_id,
        gradient_clipping=0.3,
        _version=46,
    )
    train_eval = TrainEval(
        train_metrics=[
            create_metric(accuracy),
            create_metric(calibration_error),
            create_metric(generative_single_token_loss),
            create_metric(generative_next_token_loss),
        ],
        validation_metrics=[
            create_metric(accuracy),
            create_metric(calibration_error),
            create_metric(generative_single_token_loss),
            create_metric(generative_next_token_loss),
        ],
        data_visualizer=None,
    )
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=1, num_gpus=1),
        train_config=train_config,
        train_eval=train_eval,
        epochs=ens_train_config.epochs,
        save_nth_epoch=1,
        keep_epoch_checkpoints=True,
        validate_nth_epoch=1,
    )
    return train_run