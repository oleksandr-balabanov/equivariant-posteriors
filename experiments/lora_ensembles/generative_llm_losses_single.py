import torch.nn.functional as F
import torch


# Define Loss Function for Conventional Next Token Generative Task
def generative_loss(
        logits, 
        input_ids,
        attention_mask
    ):

    # Set input_ids for masked tokens to -100 so they are not used in loss computation
    input_ids[attention_mask == 0] = -100

    # labels
    labels = input_ids.clone()
    labels[:, :-1] = input_ids.clone()[:, 1:]
    labels[:, -1] = -100  # Ignore the loss for the last token
    
    
    # Reshape logits to [batch_size * sequence_length, num_classes]
    # Reshape labels to [batch_size * sequence_length]
    loss_batch = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

    return loss_batch

def generative_next_token_loss(output, batch):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    logits = output["logits"]
    return generative_loss(
        logits, 
        input_ids, 
        attention_mask
    )

def generative_single_token_loss(
        output, 
        batch, 
        target_token_position_wrt_attention_mask=-2
    ):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    logits = output["logits"]
    single_token_attention_mask = create_single_token_attention_mask(
        attention_mask, 
        target_token_position_wrt_attention_mask
    )
    return generative_loss(
        logits, 
        input_ids, 
        single_token_attention_mask
    )

def generative_single_token_and_lora_l2(output, batch):
    return generative_single_token_loss(output, batch) + output["lora_l2_loss"]

def generative_next_token_and_lora_l2(output, batch):
    return generative_next_token_loss(output, batch) + output["lora_l2_loss"]


def calculate_answer_indices(
        attention_mask, 
        target_token_position_wrt_attention_mask
    ):
    # Create a mask where each element is True if it's zero and False otherwise
    zero_mask = attention_mask == 0

    # Convert the boolean mask to an integer mask
    int_zero_mask = zero_mask.int()

    # Find the index of the first zero in each row of the attention mask
    # If no zero is found in a row, the index will be equal to the number of columns
    first_zero_indices = int_zero_mask.argmax(dim=1, keepdim=True)

    # Handle cases where no zero was found by setting those indices to the length of the sequence
    no_zero_found = ~zero_mask.any(dim=1, keepdim=True)
    first_zero_indices[no_zero_found] = attention_mask.size(1)

    # Calculate the answer indices
    answer_indices = first_zero_indices + target_token_position_wrt_attention_mask

    return answer_indices.squeeze(1)

def create_single_token_attention_mask(
        attention_mask, 
        target_token_position_wrt_attention_mask
    ):

    # Find indices of the answer token in the attention mask of the batch.
    answer_indices = calculate_answer_indices(
        attention_mask, 
        target_token_position_wrt_attention_mask
    )

    # Initialize a new attention mask with zeros
    single_token_attention_mask = torch.zeros_like(attention_mask)

    # Create a range tensor representing batch indices
    batch_indices = torch.arange(attention_mask.size(0), device=attention_mask.device)

    # Use advanced indexing to set the answer indices to 1
    single_token_attention_mask[batch_indices, answer_indices] = 1

    return single_token_attention_mask