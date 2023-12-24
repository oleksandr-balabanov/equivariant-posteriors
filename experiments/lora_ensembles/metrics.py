import torch
import torchmetrics as tm
import torch.nn.functional as F
from typing import Dict

from experiments.lora_ensembles.generative_llm_losses import (
    create_single_token_attention_mask
)

def create_metric_sample(
    input_ids,
    attention_mask,
    logits
):
    # Set input_ids for masked tokens to -100 so they are not used in loss computation
    input_ids[attention_mask == 0] = -100

    # labels
    labels = input_ids.clone()
    labels[:, :-1] = input_ids.clone()[:, 1:]
    labels[:, -1] = -100

    # reshape to [batch_size * sequence_length, num_classes] and [batch_size * sequence_length]
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)

    # Create a mask for filtering out ignored indices
    mask = labels != -100

    # Apply the mask to filter both predictions and targets
    logits = logits[mask]
    labels = labels[mask]

    return dict(
        output=logits.detach(),
        prediction=F.softmax(logits, dim=-1).detach(),
        target=labels.detach(),
    )


def create_metric_sample_next_token_task(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    logits = output["logits"].view(-1, output["logits"].size(-1))

    metric_sample = create_metric_sample(input_ids, attention_mask, logits)

    return metric_sample


def create_metric_sample_single_token_task(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    logits = output["logits"].view(-1, output["logits"].size(-1))

    single_token_attention_mask = create_single_token_attention_mask(
        attention_mask, 
        target_token_position_wrt_attention_mask=-2,
    )

    metric_sample = create_metric_sample(input_ids, single_token_attention_mask, logits)

    return metric_sample


def create_metric_sample_single_token_task(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
):   
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Find the index of the first zero in the attention mask of the first example in the batch.
    # This (index - 2) is assumed to be the position of the "answer token" in the sequence.
    index_of_zero = (attention_mask[0] == 0).nonzero(as_tuple=True)[0]
    if len(index_of_zero) > 0:
        answer_index = index_of_zero[0].item() - 2
    else:
        # Default to the last token before padding if no zero is found.
        answer_index = -2

    # Extract logits from the model output corresponding to the answer token
    logits = output["logits"][:, answer_index - 1, :]
    labels = input_ids[:, answer_index]

    return dict(
        output=logits.detach(),
        prediction=F.softmax(logits, dim=-1).detach(),
        target=labels.detach(),
    )


def calibration_error(output, batch, create_metric_sample=create_metric_sample_single_token_task):
    num_classes = output["logits"].shape[-1]
    metric_sample = create_metric_sample(output, batch)
    return tm.functional.classification.calibration_error(
        metric_sample["prediction"],
        metric_sample["target"],
        n_bins=15,
        num_classes=num_classes,
        task="multiclass",
    )

def accuracy(output, batch, create_metric_sample=create_metric_sample_single_token_task):
    num_classes = output["logits"].shape[-1]
    metric_sample = create_metric_sample(output, batch)
    return tm.functional.accuracy(
        metric_sample["prediction"],
        metric_sample["target"],
        task="multiclass",
        num_classes=num_classes,
    )