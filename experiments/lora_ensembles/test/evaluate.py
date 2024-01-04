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
    create_metric_sample_single_token,
    create_metric_sample_next_token
)

import torch
from sklearn.metrics import roc_auc_score

def calculate_ood_performance_auroc(in_domain_softmax_probs, out_domain_softmax_probs, batch_size=1000):
    """
    Calculate the OOD detection performance measured by AUROC for PyTorch tensors.

    Args:
    in_domain_softmax_probs (torch.Tensor): Max softmax probabilities for in-domain dataset.
    out_domain_softmax_probs (torch.Tensor): Max softmax probabilities for out-of-domain dataset.
    batch_size (int): Size of batches to process, to handle large datasets.

    Returns:
    float: AUROC score indicating OOD detection performance.
    """
    # Negative max probabilities
    neg_max_probs_in_domain = -in_domain_softmax_probs
    neg_max_probs_out_domain = -out_domain_softmax_probs

    # Labels
    labels_in_domain = torch.zeros_like(neg_max_probs_in_domain)
    labels_out_domain = torch.ones_like(neg_max_probs_out_domain)

    # Concatenate probabilities and labels
    all_probs = torch.cat([neg_max_probs_in_domain, neg_max_probs_out_domain])
    all_labels = torch.cat([labels_in_domain, labels_out_domain])

    # Convert to numpy in batches and calculate AUROC
    auroc_scores = []
    for i in range(0, len(all_probs), batch_size):
        batch_probs = all_probs[i:i+batch_size].cpu().numpy()
        batch_labels = all_labels[i:i+batch_size].cpu().numpy()
        auroc_scores.append(roc_auc_score(batch_labels, batch_probs))

    # Average the AUROC scores over all batches
    avg_auroc_score = sum(auroc_scores) / len(auroc_scores)

    return avg_auroc_score

def calculate_softmax_probs_ensemble(outputs_list, batch, metric_sample_creator=create_metric_sample_single_token):
    """
    Calculate softmax probabilities for an ensemble of models.
    """
    softmax_probs_ensemble = []

    for output in outputs_list:
        metric_sample = metric_sample_creator(output, batch)
        softmax_probs_ensemble.append(metric_sample["predictions"])

    return softmax_probs_ensemble

def calculate_mean_softmax_probs(softmax_probs_ensemble):
    """
    Calculate the mean of softmax probabilities from an ensemble.
    """
    mean_softmax_probs = sum(softmax_probs_ensemble) / len(softmax_probs_ensemble)
    return mean_softmax_probs

def calculate_ensemble_accuracies(outputs_list, batch, metric_sample_creator=create_metric_sample_single_token):
    """
    Calculate accuracies for an ensemble of models.
    """
    accuracies_ensemble = []

    for output in outputs_list:
        metric_sample = metric_sample_creator(output, batch)
        predicted_tokens = torch.argmax(metric_sample["output"], dim=-1)
        true_labels = metric_sample["targets"]

        correct_predictions = torch.sum(predicted_tokens == true_labels)
        accuracy = correct_predictions.item() / len(true_labels)
        accuracies_ensemble.append(accuracy)

    return accuracies_ensemble

def calculate_accuracy_from_mean_softmax_probs(mean_softmax_probs, batch):
    """
    Calculate accuracy from mean softmax probabilities.
    """
    true_labels = batch["labels"]
    predicted_labels = torch.argmax(mean_softmax_probs, dim=-1)

    correct_predictions = torch.sum(predicted_labels == true_labels)
    accuracy = correct_predictions.item() / len(true_labels)

    return accuracy


def evaluate(lora_ensemble, eval_dataset, device):
    lora_ensemble.model.eval()
    eval_loader = DataLoader(eval_dataset, batch_size=8)

    total_ensemble_accuracy = 0
    total_individual_accuracies = [0] * len(lora_ensemble.models)
    total_samples = 0

    with torch.no_grad():
        for batch in eval_loader:
            reshaped_batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }

            outputs_list = lora_ensemble.ensemble_forward(batch=reshaped_batch)
            softmax_probs_ensemble = calculate_softmax_probs_ensemble(outputs_list, reshaped_batch)
            mean_softmax_probs = calculate_mean_softmax_probs(softmax_probs_ensemble)
            ensemble_accuracy = calculate_accuracy_from_mean_softmax_probs(mean_softmax_probs, batch)
            total_ensemble_accuracy += ensemble_accuracy * reshaped_batch["input_ids"].size(0)

            for i, output in enumerate(outputs_list):
                acc = accuracy(output, batch)
                total_individual_accuracies[i] += acc * reshaped_batch["input_ids"].size(0)

            total_samples += reshaped_batch["input_ids"].size(0)

        average_ensemble_accuracy = total_ensemble_accuracy / total_samples
        average_individual_accuracies = [acc / total_samples for acc in total_individual_accuracies]

        print("Ensemble Mean Accuracy:", average_ensemble_accuracy.item())
        for i, acc in enumerate(average_individual_accuracies):
            print(f"Model {i} Accuracy: {acc.item()}")



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
        max_len=150,
        dataset_split="train",
    )
    eval_dataset = DataCommonsenseQa(eval_dataset_config)
    evaluate(lora_ensemble, eval_dataset, device)
