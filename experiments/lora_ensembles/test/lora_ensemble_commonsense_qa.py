#!/usr/bin/env python
import torch

# import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

from generative_llm_losses import (
    generative_next_token_and_lora_l2,
    generative_next_token_loss,
    # generative_single_token_loss,
)

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import TrainEval
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.ensemble import create_ensemble_config
from lib.ensemble import request_ensemble
from lib.models.llama2generative import LLaMA2GenerativeConfig
from lib.metric import create_metric
from lib.files import prepare_results

from lib.distributed_trainer import distributed_train

# from lib.generic_ablation import generic_ablation

# import lib.data_factory as data_factory
# import lib.model_factory as model_factory
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.data_registry import DataMMLUConfig, DataMMLU
from lib.data_registry import DataCommonsenseQaConfig, DataCommonsenseQa
from lib.models.llama2generative import LLaMA2GenerativeConfig
from lib.metric import create_metric
from lib.files import prepare_results

from experiments.lora_ensembles.metrics import accuracy, calibration_error


# LLaMA Checkpoint
LLaMA_CHECKPOINT = "meta-llama/Llama-2-7b-hf"


# Configuration for Training
def create_config(
    ensemble_id,
    checkpoint=LLaMA_CHECKPOINT,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.0,
    lora_l2=0,
    regular_l2=0,
    target_modules=["q_proj", "v_proj"],
):
    train_config = TrainConfig(
        model_config=LLaMA2GenerativeConfig(
            checkpoint=checkpoint,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_l2=lora_l2,
            target_modules=target_modules,
        ),
        train_data_config=DataCommonsenseQaConfig(
            dataset="commonsense_qa",
            model_checkpoint=LLaMA_CHECKPOINT,
            max_len=100,
            dataset_split="train",
            num_samples=1,
            #subset_names=['high_school_european_history'],
        ),
        val_data_config=DataCommonsenseQaConfig(
            dataset="commonsense_qa",
            model_checkpoint=LLaMA_CHECKPOINT,
            max_len=100,
            dataset_split="train",
            num_samples=1,
            #subset_names=['high_school_european_history'],
        ),
        loss=generative_next_token_and_lora_l2,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=regular_l2, lr=1e-4),
        ),
        batch_size=1,
        ensemble_id=ensemble_id,
        gradient_clipping=0.3,
        _version=38,
    )
    train_eval = TrainEval(
        train_metrics=[
            create_metric(accuracy),
            create_metric(calibration_error),
            create_metric(generative_next_token_and_lora_l2),
            create_metric(generative_next_token_loss),
            create_metric(lambda output, batch: output["lora_l2_loss"], name="lora_l2"),
        ],
        validation_metrics=[
            create_metric(accuracy),
            create_metric(calibration_error),
            create_metric(generative_next_token_and_lora_l2),
            create_metric(generative_next_token_loss),
            create_metric(lambda output, batch: output["lora_l2_loss"], name="lora_l2"),
        ],
        data_visualizer=None,
    )
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=True, num_workers=2, num_gpus=2),
        train_config=train_config,
        train_eval=train_eval,
        epochs=100,
        save_nth_epoch=1,
        validate_nth_epoch=1,
    )
    return train_run


def evaluate(model, eval_dataset, print_sample=True):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0
    tokenizer = eval_dataset.tokenizer

    device = next(model.parameters()).device
    eval_loader = DataLoader(eval_dataset, batch_size=4)

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            labels[:, :-1] = labels.clone()[:, 1:]

            # Mask for filtering labels and logits
            valid_labels_mask = labels != -100

            reshaped_batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            outputs = model(batch=reshaped_batch)
            logits = outputs["logits"]

            # Apply mask to logits and labels
            filtered_logits = logits[valid_labels_mask]
            filtered_labels = labels[valid_labels_mask]

            loss = generative_next_token_loss(outputs, reshaped_batch)

            total_loss += loss.item()
            total_correct += (
                (filtered_logits.argmax(dim=-1) == filtered_labels).sum().item()
            )
            total_examples += filtered_labels.numel()

            if print_sample:
                # Print one sample token-by-token (from the first batch only)
                print_sample_comparison(input_ids[0], logits[0], labels[0], tokenizer)

    avg_loss = total_loss / len(eval_loader)
    accuracy = total_correct / total_examples
    print(f"Evaluation Results: Average Loss = {avg_loss}, Accuracy = {accuracy}")


def print_sample_comparison(input_ids, logits, labels, tokenizer):
    print("Sample Token-by-Token Comparison:")
    for j in range(input_ids.size(0)):
        predicted_token_id = logits[j].argmax(dim=-1).item()
        true_token_id = labels[j].item()
        if true_token_id != -100:  # Ignore padding tokens
            predicted_token = tokenizer.decode([predicted_token_id])
            true_token = tokenizer.decode([true_token_id])
            print(
                f"Token {j}: True token id: '{true_token_id}', True: '{true_token}', Predicted: '{predicted_token}'"
            )


def main():
    print("Start")
    ensemble_config = create_ensemble_config(create_config, 1)
    prepare_results("lora_ensemble", ensemble_config.members)
    print("ensemble_config finished")
    request_ensemble(ensemble_config)
    distributed_train(ensemble_config.members)
    print("ensemble finished")


if __name__ == "__main__":
    main()
