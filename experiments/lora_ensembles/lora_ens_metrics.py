import torch
import torchmetrics as tm
import torch.nn.functional as F
from typing import Dict, Callable
from torch import Tensor

from experiments.lora_ensembles.generative_llm_losses import (
    create_single_token_attention_mask
)

# Constants
IGNORE_INDEX: int = -100

def create_metric_sample_general(input_ids: Tensor, attention_mask: Tensor, logits: Tensor) -> Dict[str, Tensor]:
    """
    Create a metric sample for general tasks using input ids, attention mask, and logits.

    Args:
    - input_ids (Tensor): Input ids tensor.
    - attention_mask (Tensor): Attention mask tensor.
    - logits (Tensor): Logits tensor.

    Returns:
    - Dict[str, Tensor]: A dictionary containing outputs, predictions, and targets for metrics calculation.
    """
    # Mask input_ids for masked tokens
    input_ids[attention_mask == 0] = IGNORE_INDEX

    # Prepare labels
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = IGNORE_INDEX

    # Reshape logits and labels
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)

    # Create a mask to filter out ignored indices
    valid_indices_mask = labels != IGNORE_INDEX
    logits = logits[valid_indices_mask]
    labels = labels[valid_indices_mask]

    return {
        'output': logits.detach(),
        'predictions': F.softmax(logits, dim=-1).detach(),
        'targets': labels.detach(),
    }

def create_metric_sample_next_token(output: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Create a metric sample for next token prediction tasks.

    Args:
    - output (Dict[str, Tensor]): Output from the model.
    - batch (Dict[str, Tensor]): Batch containing 'input_ids' and 'attention_mask'.

    Returns:
    - Dict[str, Tensor]: A dictionary for metrics calculation.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    logits = output["logits"].view(-1, output["logits"].size(-1))

    return create_metric_sample_general(input_ids, attention_mask, logits)

def create_metric_sample_single_token(output: Dict[str, Tensor], batch: Dict[str, Tensor], target_token_position: int = -2) -> Dict[str, Tensor]:
    """
    Create a metric sample for single token prediction tasks.

    Args:
    - output (Dict[str, Tensor]): Output from the model.
    - batch (Dict[str, Tensor]): Batch containing 'input_ids' and 'attention_mask'.
    - target_token_position (int): Target position for the token relative to the end of the sequence.

    Returns:
    - Dict[str, Tensor]: A dictionary for metrics calculation.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    logits = output["logits"].view(-1, output["logits"].size(-1))

    single_token_attention_mask = create_single_token_attention_mask(
        attention_mask, 
        target_token_position,
    )

    return create_metric_sample_general(input_ids, single_token_attention_mask, logits)

def calibration_error(output: Dict[str, Tensor], batch: Dict[str, Tensor], metric_sample_creator: Callable = create_metric_sample_single_token) -> Tensor:
    """
    Calculate the calibration error for the given output and batch.

    Args:
    - output (Dict[str, Tensor]): Output from the model.
    - batch (Dict[str, Tensor]): Batch containing 'input_ids' and 'attention_mask'.
    - metric_sample_creator (Callable): Function to create metric samples.

    Returns:
    - Tensor: Calculated calibration error.
    """
    num_classes = output["logits"].shape[-1]
    metric_sample = metric_sample_creator(output, batch)
    return tm.functional.classification.calibration_error(
        metric_sample["predictions"],
        metric_sample["targets"],
        n_bins=10,
        num_classes=num_classes,
        task="multiclass",
    )

def accuracy(output: Dict[str, Tensor], batch: Dict[str, Tensor], metric_sample_creator: Callable = create_metric_sample_single_token) -> Tensor:
    """
    Calculate the accuracy for the given output and batch.

    Args:
    - output (Dict[str, Tensor]): Output from the model.
    - batch (Dict[str, Tensor]): Batch containing 'input_ids' and 'attention_mask'.
    - metric_sample_creator (Callable): Function to create metric samples.

    Returns:
    - Tensor: Calculated accuracy.
    """
    num_classes = output["logits"].shape[-1]
    metric_sample = metric_sample_creator(output, batch)
    return tm.functional.accuracy(
        metric_sample["predictions"],
        metric_sample["targets"],
        task="multiclass",
        num_classes=num_classes,
    )
