import torch

# import torch.nn.functional as F
from torch.utils.data import DataLoader
from lib.ddp import ddp_setup

from experiments.lora_ensembles.generative_llm_losses import (
    generative_next_token_loss,
    generative_single_token_loss,
)
from experiments.lora_ensembles.metrics import accuracy, calibration_error
from experiments.lora_ensembles.generative_llm_losses import (
    generative_next_token_and_lora_l2,
    generative_next_token_loss,
    generative_single_token_and_lora_l2,
    generative_single_token_loss,
)

from lib.ensemble import create_ensemble_config
from lib.data_registry import DataCommonsenseQaConfig, DataCommonsenseQa


from experiments.lora_ensembles.lora_inference import create_lora_ensemble, LORAEnsemble
from experiments.lora_ensembles.lora_ensemble import create_config, LLaMA_CHECKPOINT


import torch
import torch.nn.functional as F
from experiments.lora_ensembles.generative_llm_losses import (
    create_single_token_attention_mask
)
from experiments.lora_ensembles.metrics import (
    create_metric_sample_single_token_task,
    create_metric_sample_next_token_task
)

def calculate_softmax_probs_ensemble(outputs_list, batch, create_metric_sample=create_metric_sample_single_token_task):
    softmax_probs_ensemble = []

    for output in outputs_list:
        metric_sample = create_metric_sample(output, batch)
        
        # Extract softmax probabilities
        softmax_probs = metric_sample["prediction"]
        softmax_probs_ensemble.append(softmax_probs)

    return softmax_probs_ensemble


def calculate_mean_of_softmax_probs(softmax_probs_ensemble):
    # Calculate the mean of the softmax probabilities
    mean_softmax_probs = sum(softmax_probs_ensemble) / len(softmax_probs_ensemble)
    return mean_softmax_probs


def calculate_mean_probabilities(softmax_probabilities):
    """
    Calculate the mean probabilities across all ensemble members.

    :param softmax_probabilities: List of softmax probabilities for each model.
    :return: Tensor of mean probabilities.
    """
    # Stack the probabilities to create a 3D tensor [num_models, batch_size, num_classes]
    stacked_probabilities = torch.stack(softmax_probabilities)
    # Calculate the mean across the first dimension (num_models)
    mean_probabilities = torch.mean(stacked_probabilities, dim=0)
    return mean_probabilities

def calculate_member_accuracies(outputs_list, targets):
    """
    Calculate accuracy for each model in the ensemble.

    :param outputs_list: List of outputs from the ensemble models.
    :param targets: Ground truth labels.
    :return: List of accuracies for each model.
    """
    accuracies = []
    for output in outputs_list:
        logits = output["logits"]
        preds = torch.argmax(logits, dim=-1)
        corrects = preds.eq(targets.view_as(preds))
        accuracy = corrects.sum().item() / corrects.nelement()
        accuracies.append(accuracy)
    return accuracies

def calculate_ensemble_mean_softmax_accuracy(outputs_list, targets):
    """
    Calculate accuracy using the mean softmax probabilities over the ensemble members.

    :param outputs_list: List of outputs from the ensemble models.
    :param targets: Ground truth labels.
    :return: Accuracy using mean softmax probabilities.
    """
    # Calculate softmax probabilities for each model
    softmax_probabilities = [F.softmax(output["logits"], dim=-1) for output in outputs_list]
    # Stack and compute the mean softmax probabilities
    mean_softmax_probs = torch.mean(torch.stack(softmax_probabilities), dim=0)
    # Predictions are the argmax of the mean softmax probabilities
    mean_preds = torch.argmax(mean_softmax_probs, dim=-1)
    corrects = mean_preds.eq(targets.view_as(mean_preds))
    accuracy = corrects.sum().item() / corrects.nelement()
    return accuracy





def evaluate(lora_ensemble: LORAEnsemble, eval_dataset, device, print_sample=True):
    lora_ensemble.model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0
    tokenizer = eval_dataset.tokenizer

    # device = next(model.parameters()).device
    eval_loader = DataLoader(eval_dataset, batch_size=8)

    with torch.no_grad():
        total_accuracy = 0
        total_calibration_error = 0
        total_generative_loss = 0
        total_combined_loss = 0
        total_samples = 0
        i_batch = 0

        for batch in eval_loader:
            print("Batch: ", i_batch)
            i_batch+=1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_size = input_ids.size(0)

            reshaped_batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            outputs_list = lora_ensemble.ensemble_forward(batch=reshaped_batch)

            for output in outputs_list:
                output = {"logits": output["logits"].to(device), "lora_l2_loss": output.get("lora_l2_loss", 0).to(device)}

                # Ensuring all relevant tensors in the batch are on the same device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Calculate metrics and losses
                acc = accuracy(output, batch)
                cal_error = calibration_error(output, batch)
                generative_loss = generative_single_token_loss(output, batch)
                combined_loss = generative_single_token_and_lora_l2(output, batch)

                total_accuracy += acc * batch_size
                total_calibration_error += cal_error * batch_size
                total_generative_loss += generative_loss * batch_size
                total_combined_loss += combined_loss * batch_size
                total_samples += batch_size

        average_accuracy = total_accuracy / total_samples
        average_calibration_error = total_calibration_error / total_samples
        average_generative_loss = total_generative_loss / total_samples
        average_combined_loss = total_combined_loss / total_samples

        print("Average Accuracy:", average_accuracy.item())
        print("Average Calibration Error:", average_calibration_error.item())
        print("Average Generative Loss:", average_generative_loss.item())
        print("Average Combined Loss with LoRA L2:", average_combined_loss.item())


def ensemble_accuracy()
    
def ensemble_probabilities

def generative_single_token_loss


def ood(lora_ensemble: LORAEnsemble, eval_dataset_in, eval_dataset_out, device, print_sample=True):
    lora_ensemble.model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0
    tokenizer = eval_dataset.tokenizer

    eval_loader = DataLoader(eval_dataset_in, batch_size=8)
    eval_loader = DataLoader(eval_dataset_out, batch_size=8)

    with torch.no_grad():
        total_accuracy = 0
        total_calibration_error = 0
        total_generative_loss = 0
        total_combined_loss = 0
        total_samples = 0
        i_batch = 0

        for batch in eval_loader:
            print("Batch: ", i_batch)
            i_batch+=1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_size = input_ids.size(0)

            reshaped_batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            } 

            outputs_list_2 = lora_ensemble.ensemble_forward(batch=reshaped_batch)


            for output in outputs_list:
                output = {"logits": output["logits"].to(device), "lora_l2_loss": output.get("lora_l2_loss", 0).to(device)}

                # Ensuring all relevant tensors in the batch are on the same device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Calculate metrics and losses
                acc = accuracy(output, batch)
                cal_error = calibration_error(output, batch)
                generative_loss = generative_single_token_loss(output, batch)
                combined_loss = generative_single_token_and_lora_l2(output, batch)

                total_accuracy += acc * batch_size
                total_calibration_error += cal_error * batch_size
                total_generative_loss += generative_loss * batch_size
                total_combined_loss += combined_loss * batch_size
                total_samples += batch_size

        average_accuracy = total_accuracy / total_samples
        average_calibration_error = total_calibration_error / total_samples
        average_generative_loss = total_generative_loss / total_samples
        average_combined_loss = total_combined_loss / total_samples

        print("Average Accuracy:", average_accuracy.item())
        print("Average Calibration Error:", average_calibration_error.item())
        print("Average Generative Loss:", average_generative_loss.item())
        print("Average Combined Loss with LoRA L2:", average_combined_loss.item())


def create_inference_config(ensemble_id):
    config = create_config(ensemble_id, epochs=1)
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
        max_len=150,
        dataset_split="train",
    )
    eval_dataset = DataCommonsenseQa(eval_dataset_config)
    evaluate(lora_ensemble, eval_dataset, device)
