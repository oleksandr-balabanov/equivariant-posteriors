import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict

# Constants
IGNORE_INDEX: int = -100

def generative_loss(logits: Tensor, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Compute the loss for a generative model.

    Args:
    - logits (Tensor): Logits from the model, shape [batch_size, sequence_length, num_classes].
    - input_ids (Tensor): Input ids, shape [batch_size, sequence_length].
    - attention_mask (Tensor): Attention mask, shape [batch_size, sequence_length].

    Returns:
    - Tensor: Computed loss.
    """
    if not all(x.dim() == 2 for x in [input_ids, attention_mask]) or logits.dim() != 3:
        raise ValueError("Input tensors must have the correct shape.")
    
    input_ids_masked = input_ids.masked_fill(attention_mask == 0, IGNORE_INDEX)
    labels = input_ids_masked.roll(-1, dims=1)
    labels[:, -1] = IGNORE_INDEX
    loss_batch = F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=IGNORE_INDEX
    ) 

    return loss_batch

def generative_next_token_loss(output: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
    """
    Calculate the loss for the next token generative task.

    Args:
    - output (Dict[str, Tensor]): Output from the model containing logits.
    - batch (Dict[str, Tensor]): Batch containing 'input_ids' and 'attention_mask'.

    Returns:
    - Tensor: Computed loss.
    """
    try:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        logits = output["logits"]
    except KeyError:
        raise ValueError("Batch and output must contain 'input_ids', 'attention_mask', and 'logits'.")

    return generative_loss(logits, input_ids, attention_mask)

def generative_single_token_loss(output: Dict[str, Tensor], batch: Dict[str, Tensor], target_token_position: int = -2) -> Tensor:
    """
    Calculate the loss for a single token in the generative task.

    Args:
    - output (Dict[str, Tensor]): Output from the model containing logits.
    - batch (Dict[str, Tensor]): Batch containing 'input_ids' and 'attention_mask'.
    - target_token_position (int): The target position for the token relative to the end of the sequence.

    Returns:
    - Tensor: Computed loss.
    """
    try:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        logits = output["logits"]
    except KeyError:
        raise ValueError("Batch and output must contain 'input_ids', 'attention_mask', and 'logits'.")

    single_token_mask = create_single_token_attention_mask(
        attention_mask, target_token_position
    )

    return generative_loss(logits, input_ids, single_token_mask)

def generative_single_token_and_lora_l2(output: Dict[str, Tensor], batch: Dict[str, Tensor], target_token_position: int = -2) -> Tensor:
    """
    Calculate the combined loss of a single token generative task and Lora L2 loss.

    Args:
    - output (Dict[str, Tensor]): Output from the model containing logits and 'lora_l2_loss'.
    - batch (Dict[str, Tensor]): Batch containing 'input_ids' and 'attention_mask'.
    - target_token_position (int): The target position for the token relative to the end of the sequence.

    Returns:
    - Tensor: Computed combined loss.
    """
    return generative_single_token_loss(output, batch, target_token_position) + output["lora_l2_loss"]

def generative_next_token_and_lora_l2(output: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
    """
    Calculate the combined loss of the next token generative task and Lora L2 loss.

    Args:
    - output (Dict[str, Tensor]): Output from the model containing logits and 'lora_l2_loss'.
    - batch (Dict[str, Tensor]): Batch containing 'input_ids' and 'attention_mask'.

    Returns:
    - Tensor: Computed combined loss.
    """
    return generative_next_token_loss(output, batch) + output["lora_l2_loss"]

def calculate_answer_indices(attention_mask: Tensor, target_token_position: int) -> Tensor:
    """
    Calculate the indices of the target token based on the attention mask.

    Args:
    - attention_mask (Tensor): Attention mask, shape [batch_size, sequence_length].
    - target_token_position (int): Position of the target token relative to the end of the sequence.

    Returns:
    - Tensor: Indices of the target token.
    """
    if attention_mask.dim() != 2:
        raise ValueError("Attention mask must be a 2D tensor.")


    zero_mask = attention_mask == 0
    int_zero_mask = zero_mask.int()
    first_zero_indices = int_zero_mask.argmax(dim=1, keepdim=True)
    no_zero_found = ~zero_mask.any(dim=1, keepdim=True)
    first_zero_indices[no_zero_found] = attention_mask.size(1)
    answer_indices = first_zero_indices + target_token_position

    return answer_indices.squeeze(1)

def create_single_token_attention_mask(attention_mask: Tensor, target_token_position: int) -> Tensor:
    """
    Create an attention mask for a single token in the sequence.

    Args:
    - attention_mask (Tensor): Original attention mask, shape [batch_size, sequence_length].
    - target_token_position (int): Position of the target token relative to the end of the sequence.

    Returns:
    - Tensor: Modified attention mask focusing on a single token.
    """
    answer_indices = calculate_answer_indices(
        attention_mask, target_token_position
    )
    single_token_attention_mask = torch.zeros_like(attention_mask)
    batch_indices = torch.arange(attention_mask.size(0), device=attention_mask.device)
    single_token_attention_mask[batch_indices, answer_indices] = 1

    return single_token_attention_mask
