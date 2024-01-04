import torch
import torchmetrics as tm
import torch.nn.functional as F
from typing import Dict

from experiments.lora_ensembles.generative_llm_losses_single import (
    create_single_token_attention_mask
)

def create_metric_sample_general(input_ids, attention_mask, logits):
    """
    Create a metric sample for general tasks by processing input ids, attention mask, and logits.
    """
    # Mask input_ids for masked tokens
    input_ids[attention_mask == 0] = -100 

    # Prepare labels
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100

    # Reshape logits and labels for loss computation
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)

    # Create a mask to filter out ignored indices
    valid_indices_mask = labels != -100
    logits = logits[valid_indices_mask]
    labels = labels[valid_indices_mask]

    return {
        'output': logits.detach(),
        'predictions': F.softmax(logits, dim=-1).detach(),
        'targets': labels.detach(),
    }

def create_metric_sample_next_token(output, batch):
    """
    Create a metric sample for next token prediction tasks.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    logits = output["logits"].view(-1, output["logits"].size(-1))

    return create_metric_sample_general(input_ids, attention_mask, logits)

def create_metric_sample_single_token(output, batch, target_token_position=-2):
    """
    Create a metric sample for single token prediction tasks.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    logits = output["logits"].view(-1, output["logits"].size(-1))

    single_token_attention_mask = create_single_token_attention_mask(
        attention_mask, 
        target_token_position,
    )

    return create_metric_sample_general(input_ids, single_token_attention_mask, logits)



def calibration_error(output, batch, metric_sample_creator=create_metric_sample_single_token):
    num_classes = output["logits"].shape[-1]
    metric_sample = metric_sample_creator(output, batch)
    return tm.functional.classification.calibration_error(
        metric_sample["predictions"],
        metric_sample["targets"],
        n_bins=15,
        num_classes=num_classes,
        task="multiclass",
    )

def accuracy(output, batch, metric_sample_creator=create_metric_sample_single_token):
    num_classes = output["logits"].shape[-1]
    metric_sample = metric_sample_creator(output, batch)
    return tm.functional.accuracy(
        metric_sample["predictions"],
        metric_sample["targets"],
        task="multiclass",
        num_classes=num_classes,
    )