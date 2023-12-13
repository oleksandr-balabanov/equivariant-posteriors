#!/usr/bin/env python
import torch
import torch.nn.functional as F
import os

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.classification_metrics import create_classification_metrics
from lib.generic_ablation import generic_ablation

import lib.data_factory as data_factory
import lib.model_factory as model_factory
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.data_registry import DataCommonsenseQaConfig
from lib.models.llama2generative import LLama2GenerativeConfig



# Define Loss Function for Generative Task
def generative_loss(outputs, batch):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']

    # Set input_ids for masked tokens to -100 so they are not used in loss computation
    input_ids[attention_mask == 0] = -100

    # labels
    labels = input_ids.clone()
    labels[:, :-1] = input_ids.clone()[:, 1:]
    labels[:, -1] = -100  # Ignore the loss for the last token

    # loss
    logits = outputs["logits"]
    # Reshape logits to [batch_size * sequence_length, num_classes]
    # Reshape labels to [batch_size * sequence_length]
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

# LLAMA Checkpoint
LLAMA_CHECKPOINT = "meta-llama/Llama-2-7b-hf"

# Configuration for Training
def create_config(ensemble_id, lora_rank=16):
    train_config = TrainConfig(
        model_config=LLama2GenerativeConfig(checkpoint=LLAMA_CHECKPOINT, lora_rank=lora_rank),
        train_data_config=DataCommonsenseQaConfig(
            dataset="commonsense_qa",
            model_checkpoint=LLAMA_CHECKPOINT,
            max_len=256,
        ),
        val_data_config=DataCommonsenseQaConfig(
            dataset="commonsense_qa",
            model_checkpoint=LLAMA_CHECKPOINT,
            max_len=256,
            validation=True,
        ),
        loss=generative_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=0.001, lr=1e-4),
        ),
        batch_size=4,
        ensemble_id=ensemble_id,
        gradient_clipping=0.3,
        _version=37,
    )
    train_eval = create_classification_metrics(None, 32000)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=20,
        save_nth_epoch=1,
        validate_nth_epoch=1,
    )
    return train_run

def main():
    print("Start")
    if torch.cuda.is_available():
        device_id = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    else:
        device_id = "cpu"
    print("device finished")
    ensemble_config = create_ensemble_config(create_config, 1)
    print("ensemble_config finished")
    ensemble = create_ensemble(ensemble_config, device_id)
    print("ensemble finished")
    print(len(ensemble))

if __name__ == "__main__":
    main()
