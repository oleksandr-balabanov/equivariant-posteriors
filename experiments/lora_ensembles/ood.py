from torch.utils.data import DataLoader
import torchmetrics as tm
import torch
import torch.nn.functional as F
from lib.ddp import ddp_setup
from experiments.lora_ensembles.metrics import (
    create_metric_sample_single_token
)
from math import ceil
from lib.ensemble import create_ensemble_config
from lib.data_registry import DataCommonsenseQaConfig, DataCommonsenseQa
from experiments.lora_ensembles.lora_inference import create_lora_ensemble, LORAEnsemble
from experiments.lora_ensembles.lora_ensemble import create_config, LLaMA_CHECKPOINT

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

def calculate_targets_ensemble(outputs_list, batch, metric_sample_creator=create_metric_sample_single_token):
    """
    Calculate targets for an ensemble of models.
    """
    targets_ensemble = []

    for output in outputs_list:
        metric_sample = metric_sample_creator(output, batch)
        targets_ensemble.append(metric_sample["targets"])

    return targets_ensemble

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


def evaluate_model_for_softmax_probs(lora_ensemble, eval_dataset, device):
    lora_ensemble.model.train()
    eval_loader = DataLoader(eval_dataset, batch_size=8)

    # Initialize lists to store tensors
    accumulated_targets = []
    accumulated_probs = []

    with torch.no_grad():
        for i_batch, batch in enumerate(eval_loader):
            if i_batch % 100 == 0:
                print("Batch: ", i_batch)

            reshaped_batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }

            outputs_list = lora_ensemble.ensemble_forward(batch=reshaped_batch)
            softmax_probs_ensemble = calculate_softmax_probs_ensemble(outputs_list, reshaped_batch)
            targets_ensemble = calculate_targets_ensemble(outputs_list, reshaped_batch)
            mean_softmax_probs = calculate_mean_softmax_probs(softmax_probs_ensemble)

            # Append tensors to lists
            accumulated_targets.append(targets_ensemble[0])
            accumulated_probs.append(mean_softmax_probs)

    # Concatenate lists of tensors into single tensors
    final_targets = torch.cat(accumulated_targets)
    final_softmax_probs = torch.cat(accumulated_probs)

    # Calculate accuracy using the accumulated tensors
    predicted_labels = torch.argmax(final_softmax_probs, dim=1)
    total_correct = torch.sum(predicted_labels == final_targets).item()
    total_examples = final_targets.size(0)
    accuracy = total_correct / total_examples
    print(f"Ensemble Accuracy over the dataset: {accuracy:.4f}")

    # Calculate and print the ensemble single token loss
    average_loss = average_generative_loss_ens(final_softmax_probs, final_targets)
    print(f"Average Ensemble Single Token Loss over the dataset: {average_loss:.4f}")

    # Calculate and print the ensemble calibration error
    ce = calibration_error(final_softmax_probs, final_targets)
    print(f"Ensemble Calibration Error over the dataset: {ce:.4f}")

    # Store the maximum softmax probabilities
    max_softmax_probs = torch.max(final_softmax_probs, dim=1).values

    return max_softmax_probs


def calibration_error(predictions, targets):
    num_classes = predictions.shape[-1]
    return tm.functional.classification.calibration_error(
        predictions,
        targets,
        n_bins=15,
        num_classes=num_classes,
        task="multiclass",
    )


def roc_auc_score_torch(y_true, y_scores):
    # Ensure tensors
    y_true = torch.tensor(y_true, dtype=torch.bool)  # Convert to boolean tensor
    y_scores = torch.tensor(y_scores, dtype=torch.float32)

    # Sort by scores
    desc_score_indices = torch.argsort(y_scores, descending=True)
    y_true = y_true[desc_score_indices]
    y_scores = y_scores[desc_score_indices]

    # Number of positive and negative examples
    n_positives = torch.sum(y_true)
    n_negatives = y_true.size(0) - n_positives

    # Calculate TPR and FPR at each threshold
    tpr = torch.cumsum(y_true, dim=0) / n_positives
    fpr = torch.cumsum(~y_true, dim=0) / n_negatives  # Now y_true is a boolean tensor

    # Adding 0 at the beginning of each for the initial point (0,0) in ROC space
    tpr = torch.cat([torch.zeros(1), tpr])
    fpr = torch.cat([torch.zeros(1), fpr])

    # Calculate the AUROC as the area under the ROC curve
    auroc = torch.trapz(tpr, fpr)

    return auroc.item()


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
    #for i in range(0, len(all_probs), batch_size):
    batch_probs = all_probs.cpu().numpy()
    batch_labels = all_labels.cpu().numpy()
    auroc_scores.append(roc_auc_score_torch(batch_labels, batch_probs))

    # Average the AUROC scores over all batches
    avg_auroc_score = sum(auroc_scores) / len(auroc_scores)

    return avg_auroc_score

def average_generative_loss_ens(softmax_probs, targets, epsilon=1e-9):

    # Convert softmax probabilities to log probabilities
    log_probs = torch.log(softmax_probs + epsilon)

    # Reshape log_probs and labels to be 2D and 1D tensors respectively
    log_probs = log_probs.view(-1, log_probs.size(-1))
    targets = targets.view(-1)

    # Compute the negative log likelihood loss
    loss_batch = F.nll_loss(log_probs, targets, reduction='mean')

    return loss_batch

def create_inference_config(ensemble_id):
    config = create_config(ensemble_id)
    config.compute_config.distributed = False
    config.compute_config.num_gpus = 1
    return config

def main():
    device = ddp_setup()
    ensemble_config = create_ensemble_config(create_inference_config, 1)
    lora_ensemble = create_lora_ensemble(ensemble_config.members, device)

    # Evaluate on in-domain dataset
    train_dataset_config = DataCommonsenseQaConfig(
        dataset="commonsense_qa",
        model_checkpoint=LLaMA_CHECKPOINT,
        max_len=150,
        dataset_split="validation",
    )
    train_dataset = DataCommonsenseQa(train_dataset_config)
    in_domain_max_probs = evaluate_model_for_softmax_probs(lora_ensemble, train_dataset, device)

    # Evaluate on out-of-domain dataset
    test_dataset_config = DataCommonsenseQaConfig(
        dataset="commonsense_qa",
        model_checkpoint=LLaMA_CHECKPOINT,
        max_len=150,
        dataset_split="train",
    )
    test_dataset = DataCommonsenseQa(test_dataset_config)
    out_domain_max_probs = evaluate_model_for_softmax_probs(lora_ensemble, test_dataset, device)

    # Calculate OOD performance
    auroc_score = calculate_ood_performance_auroc(in_domain_max_probs, out_domain_max_probs)
    print("OOD AUROC Score:", auroc_score)

if __name__ == "__main__":
    main()

