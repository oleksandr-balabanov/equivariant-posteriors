import torch

# import torch.nn.functional as F
from torch.utils.data import DataLoader
from lib.ddp import ddp_setup

from experiments.lora_ensembles.generative_llm_losses import (
    generative_next_token_loss,
    generative_single_token_loss,
)


# from lib.generic_ablation import generic_ablation

# import lib.data_factory as data_factory
# import lib.model_factory as model_factory
from lib.ensemble import create_ensemble_config
from lib.data_registry import DataCommonsenseQaConfig, DataCommonsenseQa


from experiments.lora_ensembles.lora_inference import create_lora_ensemble, LORAEnsemble
from experiments.lora_ensembles.lora_ensemble import create_config, LLaMA_CHECKPOINT


def evaluate(lora_ensemble: LORAEnsemble, eval_dataset, device, print_sample=True):
    lora_ensemble.model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0
    tokenizer = eval_dataset.tokenizer

    # device = next(model.parameters()).device
    eval_loader = DataLoader(eval_dataset, batch_size=4)

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            labels[:, :-1] = labels.clone()[:, 1:]
            #print(input_ids.shape)

            # Mask for filtering labels and logits
            #valid_labels_mask = labels != -100

            reshaped_batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            outputs_list = lora_ensemble.ensemble_forward(batch=reshaped_batch)
            print(outputs_list)
            logits = outputs_list[0]["logits"]

            target_token_position_wrt_attention_mask = -2
            index_of_zero = (attention_mask[0] == 0).nonzero(as_tuple=True)[0]
            if len(index_of_zero) > 0:
                answer_index = index_of_zero[0].item() + target_token_position_wrt_attention_mask
            else:
                # Default to the last token before padding if no zero is found.
                answer_index = target_token_position_wrt_attention_mask


            # Extracting the logits tensors into a list
            #logits_tensors = [d['logits'][valid_labels_mask] for d in outputs_list]

            # Stacking them into a single tensor along a new batch dimension
            #filtered_logits = torch.stack(logits_tensors, dim=0)

            # Apply mask to logits and labels
            #filtered_labels = labels[valid_labels_mask]

            #loss = generative_single_token_loss(outputs_list[0], reshaped_batch)

            #total_loss += loss.item()
            #print(filtered_logits.shape)
            #total_correct += (
            #    (filtered_logits.argmax(dim=-1) == filtered_labels).sum().item()
            #)
            #total_examples += filtered_labels.numel()

    if print_sample:
        # Print one sample token-by-token (from the first batch only)
        print_sample_comparison(input_ids[0], logits[0], labels[0], tokenizer, answer_index)

    avg_loss = total_loss / len(eval_loader)
    accuracy = total_correct / total_examples
    print(f"Evaluation Results: Average Loss = {avg_loss}, Accuracy = {accuracy}")


def print_sample_comparison(input_ids, logits, labels, tokenizer, answer_index):
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

    true_token_id = input_ids[answer_index].item()
    predicted_token_id = logits[answer_index-1].argmax(dim=-1).item()
    predicted_token = tokenizer.decode([predicted_token_id])
    true_token = tokenizer.decode([true_token_id])
    print(f"Answer Token: True token id: '{true_token_id}', True: '{true_token}', Predicted: '{predicted_token}'")

def create_inference_config(ensemble_id):
    config = create_config(ensemble_id)
    config.compute_config.distributed = False
    config.compute_config.num_gpus = 1
    return config


if __name__ == "__main__":
    device = ddp_setup()
    ensemble_config = create_ensemble_config(create_inference_config, 1)
    lora_ensemble = create_lora_ensemble(ensemble_config.members, device)

    eval_dataset_config = DataCommonsenseQaConfig(
        dataset="commonsense_qa",
        model_checkpoint=LLaMA_CHECKPOINT,
        max_len=256,
        dataset_split="train",
        num_samples=1
    )
    eval_dataset = DataCommonsenseQa(eval_dataset_config)
    evaluate(lora_ensemble, eval_dataset, device)
