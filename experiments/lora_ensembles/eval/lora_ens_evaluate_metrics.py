from typing import List, Dict, Callable, Tuple
import lib.data_factory as data_factory


import torch
import torchmetrics as tm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.lora_ensembles.utils.lora_ens_metrics import create_metric_sample_single_token
from experiments.lora_ensembles.utils.lora_ens_inference import LORAEnsemble
from experiments.lora_ensembles.utils.lora_ens_file_operations import save_to_dill, load_from_dill

IGNORE_INDEX = -100

def calculate_softmax_probs_ensemble(
    outputs_list: List[torch.Tensor], 
    batch: Dict[str, torch.Tensor], 
    metric_sample_creator: Callable = create_metric_sample_single_token
) -> torch.Tensor:
    """
    Calculate softmax probabilities for each output in an ensemble of models.

    Args:
        outputs_list (List[torch.Tensor]): Outputs from each model in the ensemble.
        batch (Dict[str, torch.Tensor]): Batch of input data.
        metric_sample_creator (Callable): Function to create a metric sample from the model's output.

    Returns:
        torch.Tensor: Concatenated softmax probabilities from the ensemble.
    """
    softmax_probs_ensemble = []

    for output in outputs_list:
        metric_sample = metric_sample_creator(output, batch)
        predictions = metric_sample["predictions"].unsqueeze(0)
        softmax_probs_ensemble.append(predictions)

    return torch.cat(softmax_probs_ensemble)

def calculate_mean_softmax_probs(softmax_probs_ensemble: torch.Tensor) -> torch.Tensor:
    """
    Calculate the mean of softmax probabilities from an ensemble.

    Args:
        softmax_probs_ensemble (torch.Tensor): Tensor containing softmax probabilities from the ensemble.

    Returns:
        torch.Tensor: Mean softmax probabilities.
    """
    mean_softmax_probs = torch.mean(softmax_probs_ensemble, dim=0)
    return mean_softmax_probs

def calculate_targets_ensemble(
    outputs_list: List[torch.Tensor], 
    batch: Dict[str, torch.Tensor], 
    metric_sample_creator: Callable = create_metric_sample_single_token
) -> List[torch.Tensor]:
    """
    Calculate targets for an ensemble of models.

    Args:
        outputs_list (List[torch.Tensor]): Outputs from each model in the ensemble.
        batch (Dict[str, torch.Tensor]): Batch of input data.
        metric_sample_creator (Callable): Function to create a metric sample from the model's output.

    Returns:
        List[torch.Tensor]: List of targets for each model in the ensemble.
    """
    targets_ensemble = []

    for output in outputs_list:
        metric_sample = metric_sample_creator(output, batch)
        targets_ensemble.append(metric_sample["targets"])

    return targets_ensemble

def calculate_ens_softmax_probs_and_targets(
    eval_dataset_config, 
    lora_ensemble: LORAEnsemble, 
    device: torch.device,
    save_file_path:str = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates ensemble softmax probabilities and targets for the evaluation dataset.

    Args:
        lora_ensemble (LORAEnsemble): The ensemble of LoRA models.
        eval_dataset_config: The dataset config for evaluation.
        device (torch.device): The device to perform calculations on.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Ensemble softmax probabilities and corresponding targets.
    """
    lora_ensemble.model.train()    
    eval_dataset = data_factory.get_factory().create(eval_dataset_config)
    eval_loader = DataLoader(eval_dataset, batch_size=5)
    accumulated_targets = []
    accumulated_ens_probs = []

    with torch.no_grad():
        for i_batch, batch in enumerate(eval_loader):
            if i_batch % 100 == 0:
                print("Batch: ", i_batch)
            reshaped_batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            try:
                outputs_list = lora_ensemble.ensemble_forward(batch=reshaped_batch)
                softmax_probs_ensemble = calculate_softmax_probs_ensemble(outputs_list, reshaped_batch)
                targets_ensemble = calculate_targets_ensemble(outputs_list, reshaped_batch)
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                continue

            accumulated_targets.append(targets_ensemble[0])
            accumulated_ens_probs.append(softmax_probs_ensemble)

    final_targets = torch.cat(accumulated_targets)
    final_ens_softmax_probs = torch.cat(accumulated_ens_probs, dim=1)

    if save_file_path:
        res_dic = {
           "targets": final_targets,
           "ens_softmax_probs": final_ens_softmax_probs,
        }
        save_to_dill(res_dic, save_file_path)

    return final_ens_softmax_probs, final_targets

def calculate_accuracy_over_ens(
    softmax_probs_ensemble: torch.Tensor, 
    final_targets: torch.Tensor
) -> float:
    """
    Calculate overall accuracy of the ensemble.

    Args:
        softmax_probs_ensemble (torch.Tensor): Mean softmax probabilities across the ensemble.
        final_targets (torch.Tensor): Target labels.

    Returns:
        float: Accuracy of the ensemble.
    """
    mean_softmax_probs = calculate_mean_softmax_probs(softmax_probs_ensemble)
    predicted_labels = torch.argmax(mean_softmax_probs, dim=-1)
    correct_predictions = torch.sum(predicted_labels == final_targets)
    print(predicted_labels)
    accuracy = correct_predictions.item() / len(final_targets)

    return accuracy

def calculate_generative_loss_ens(
    ens_softmax_probs: torch.Tensor, 
    ens_targets: torch.Tensor, 
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Calculate generative loss for the ensemble, focusing on specific classes.

    Args:
        ens_softmax_probs (torch.Tensor): Ensemble softmax probabilities.
        ens_targets (torch.Tensor): Ensemble targets.
        epsilon (float): Small value to prevent log(0). Default is 1e-6.

    Returns:
        torch.Tensor: Calculated loss.
    """

    # Calculate the mean softmax probabilities for the relevant classes
    mean_softmax_probs = calculate_mean_softmax_probs(ens_softmax_probs)

    # Log probabilities
    log_probs = torch.log(mean_softmax_probs + epsilon)

    # Calculate loss
    loss = F.nll_loss(log_probs, ens_targets, reduction='mean', ignore_index=IGNORE_INDEX)
    return loss.item()

def calculate_ce_over_ens(
    ens_softmax_probs: torch.Tensor, 
    ens_targets: torch.Tensor
) -> torch.Tensor:
    """
    Calculate calibration error over ensemble softmax probabilities.

    Args:
        ens_softmax_probs (torch.Tensor): Ensemble softmax probabilities.
        ens_targets (torch.Tensor): Targets for the ensemble.

    Returns:
        torch.Tensor: Calibration error.
    """
    mean_softmax_probs = calculate_mean_softmax_probs(ens_softmax_probs)
    num_classes = mean_softmax_probs.shape[-1]

    return tm.functional.classification.calibration_error(
        mean_softmax_probs,
        ens_targets,
        n_bins=10,
        num_classes=num_classes,
        task="multiclass",
    ).item()

def calculate_roc_auc_score(
    y_true: torch.Tensor, 
    y_scores: torch.Tensor
) -> float:
    """
    Calculate the ROC AUC score.

    Args:
        y_true (torch.Tensor): Ground truth binary labels.
        y_scores (torch.Tensor): Predicted scores.

    Returns:
        float: ROC AUC score.
    """
    y_true = torch.tensor(y_true, dtype=torch.bool)
    y_scores = torch.tensor(y_scores, dtype=torch.float32)

    desc_score_indices = torch.argsort(y_scores, descending=True)
    y_true = y_true[desc_score_indices]
    y_scores = y_scores[desc_score_indices]

    n_positives = torch.sum(y_true)
    n_negatives = y_true.size(0) - n_positives

    tpr = torch.cumsum(y_true, dim=0) / n_positives
    fpr = torch.cumsum(~y_true, dim=0) / n_negatives

    tpr = torch.cat([torch.zeros(1), tpr])
    fpr = torch.cat([torch.zeros(1), fpr])

    auroc = torch.trapz(tpr, fpr)

    return auroc.item()

def calculate_ood_performance_auroc(
    in_domain_scores: torch.Tensor, 
    out_domain_scores: torch.Tensor
) -> float:
    """
    Calculate OOD performance using AUROC score.

    Args:
        in_domain_scores (torch.Tensor): Scores for in-domain samples.
        out_domain_scores (torch.Tensor): Scores for out-of-domain samples.

    Returns:
        float: AUROC score representing OOD performance.
    """
    labels_in_domain = torch.zeros_like(in_domain_scores)
    labels_out_domain = torch.ones_like(out_domain_scores)

    all_scores = torch.cat([in_domain_scores, out_domain_scores])
    all_labels = torch.cat([labels_in_domain, labels_out_domain])

    all_scores = all_scores.cpu()
    all_labels = all_labels.cpu()

    auroc_score = calculate_roc_auc_score(all_labels, all_scores)

    return auroc_score

def calculate_max_average_probs(
    softmax_probs_ensemble: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the maximum of average probabilities from ensemble softmax probabilities.

    Args:
        softmax_probs_ensemble (torch.Tensor): Ensemble softmax probabilities.

    Returns:
        torch.Tensor: Maximum of average probabilities.
    """
    mean_softmax_probs = calculate_mean_softmax_probs(softmax_probs_ensemble)
    max_mean_softmax_probs, _ = torch.max(mean_softmax_probs, dim=-1)

    return max_mean_softmax_probs

def calculate_entropy(
    softmax_probs_ensemble: torch.Tensor, 
    dim: int = -1, 
    eps: float = 1e-9
) -> torch.Tensor:
    """
    Calculate entropy of softmax probabilities.

    Args:
        softmax_probs_ensemble (torch.Tensor): Softmax probabilities.
        dim (int): Dimension over which to calculate entropy. Defaults to -1.
        eps (float): Small value to prevent log(0). Defaults to 1e-9.

    Returns:
        torch.Tensor: Entropy of softmax probabilities.
    """
    softmax_probs = calculate_mean_softmax_probs(softmax_probs_ensemble)
    entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + eps), dim=dim)

    return entropy

def calculate_mutual_information(
    softmax_probs_ensemble: torch.Tensor
) -> torch.Tensor:
    """
    Calculate mutual information for an ensemble.

    Args:
        softmax_probs_ensemble (torch.Tensor): Ensemble softmax probabilities.

    Returns:
        torch.Tensor: Mutual information.
    """
    entropy_of_the_mean_ens_prob = calculate_entropy(softmax_probs_ensemble)
    mean_ens_entropy = torch.mean(calculate_entropy(softmax_probs_ensemble.unsqueeze(0)), dim=0)
    
    mutual_information = entropy_of_the_mean_ens_prob - mean_ens_entropy

    return mutual_information


def evaluate_lora_ens_on_dataset(
    dataset_config,
    lora_ensemble: LORAEnsemble,  
    device: torch.device, 
    save_file_path: str = None,
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate the LORA ensemble on a single dataset to calculate accuracy, loss, and calibration error.

    Args:
        lora_ensemble (LORAEnsemble): The ensemble of LoRA models.
        dataset config: The dataset config to evaluate on.
        device (torch.device): The device to perform calculations on.
        
    Returns:
        Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]: Accuracy, loss, calibration error, and OOD scores.
    """
    try:
        # load from file
        "Loading the ens softmax probabilitiess..."
        res = load_from_dill(save_file_path)
        ens_probs = res["ens_softmax_probs"]
        targets = res["targets"]
        "Loading is complete."
    except:
        # if fail then calculate and save
        "Loading is not complete: Calculating the ens softmax probabilities from scratch..."
        ens_probs, targets = calculate_ens_softmax_probs_and_targets(dataset_config, lora_ensemble, device, save_file_path)
        "Calculation is complete."

    # calculate the metrics
    accuracy = calculate_accuracy_over_ens(ens_probs, targets)
    loss = calculate_generative_loss_ens(ens_probs, targets)
    ce = calculate_ce_over_ens(ens_probs, targets)

    ood_scores_max_probs = calculate_max_average_probs(ens_probs)
    ood_scores_entropy = calculate_entropy(ens_probs)
    ood_scores_mi = calculate_mutual_information(ens_probs)
    ood_scores = {
        "ood_scores_max_probs":ood_scores_max_probs,
        "ood_scores_entropy":ood_scores_entropy,
        "ood_scores_mi":ood_scores_mi
    }

    print_single_dataset_results(dataset_config.dataset, accuracy, loss, ce)

    return accuracy, loss, ce, ood_scores

def print_single_dataset_results(
    dataset_name: str, 
    accuracy: float, 
    loss: torch.Tensor, 
    calibration_error: torch.Tensor
):
    """
    Print the evaluation results for a single dataset.

    Args:
        dataset_name (str): Name of the dataset.
        accuracy (float): Calculated accuracy.
        loss (torch.Tensor): Calculated loss.
        calibration_error (torch.Tensor): Calculated calibration error.
    """
    print(f"Results for {dataset_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss: {loss:.4f}")
    print(f"Calibration Error: {calibration_error:.4f}")

def evaluate_lora_ens_on_two_datasets_and_ood(
    dataset_1_config, 
    dataset_2_config, 
    lora_ensemble: LORAEnsemble, 
    device: torch.device,
    save_file_path_1: str = None,
    save_file_path_2: str = None,
) -> Dict[str, float]:
    """
    Evaluate the LORA ensemble on two distinct datasets and calculate Out-Of-Distribution (OOD) performance.
    This function assesses accuracy, loss, and calibration error on each dataset and computes the OOD score 
    between the datasets.

    Args:
        lora_ensemble (LORAEnsemble): The ensemble of LoRA models.
        dataset_1_config: The first dataset config for evaluation.
        dataset_2_config: The second dataset config for evaluation.
        device (torch.device): The device on which computations will be performed.

    Returns:
        Dict[str, float]: A dictionary containing accuracy ('acc_one', 'acc_two'), 
        loss ('loss_one', 'loss_two'), calibration error ('ce_one', 'ce_two') for each dataset, 
        and the OOD scores ('ood_score') between them.
    """
    
    # Evaluate on the first dataset
    acc_one, loss_one, ce_one, ood_scores_one = evaluate_lora_ens_on_dataset(dataset_1_config, lora_ensemble, device, save_file_path=save_file_path_1)

    # Evaluate on the second dataset
    acc_two, loss_two, ce_two, ood_scores_two = evaluate_lora_ens_on_dataset(dataset_2_config, lora_ensemble, device, save_file_path=save_file_path_2)

    # Calculate OOD performance
    ood_score_max_probs = calculate_ood_performance_auroc(ood_scores_one["ood_scores_max_probs"], ood_scores_two["ood_scores_max_probs"])
    ood_score_entropy = calculate_ood_performance_auroc(ood_scores_one["ood_scores_entropy"], ood_scores_two["ood_scores_entropy"])
    ood_score_mi = calculate_ood_performance_auroc(ood_scores_one["ood_scores_mi"], ood_scores_two["ood_scores_mi"])
    ood_score = {
        "ood_score_max_probs":ood_score_max_probs,
        "ood_score_entropy":ood_score_entropy,
        "ood_score_mi":ood_score_mi,
    }
    print_odd(ood_score, dataset_1_config.dataset, dataset_2_config.dataset)

    return {
        "acc_one": acc_one, 
        "loss_one": loss_one, 
        "ce_one": ce_one,
        "acc_two": acc_two, 
        "loss_two": loss_two, 
        "ce_two": ce_two,
        "ood_score": ood_score
    }

def print_odd(
    ood_score: Dict, 
    in_domain_dataset_name: str, 
    out_of_domain_dataset_name: str
):
    """
    Print the OOD AUROC Score.

    Args:
        ood_score (float): Calculated OOD AUROC score.
        in_domain_dataset_name (str): Name of the in-domain dataset.
        out_of_domain_dataset_name (str): Name of the out-of-domain dataset.
    """
    for ood_key in ood_score.keys():
        print(f"OOD AUROC Score for {ood_key} and for in domain {in_domain_dataset_name}/out-of-domain {out_of_domain_dataset_name}: {ood_score[ood_key]:.4f}")





   