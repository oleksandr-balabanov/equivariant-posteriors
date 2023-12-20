import torch
import torchmetrics as tm
import torch.nn.functional as F
from typing import Dict


def create_metric_sample_next_token_task(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Set input_ids for masked tokens to -100 so they are not used in loss computation
    input_ids[attention_mask == 0] = -100

    # labels
    labels = input_ids.clone()
    labels[:, :-1] = input_ids.clone()[:, 1:]
    labels[:, -1] = -100

    # reshape to [batch_size * sequence_length, num_classes] and [batch_size * sequence_length]
    logits = output["logits"].view(-1, output["logits"].size(-1))
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


def calibration_error(output, batch, create_metric_sample=create_metric_sample_next_token_task):
    num_classes = output["logits"].shape[-1]
    metric_sample = create_metric_sample(output, batch)
    return tm.functional.classification.calibration_error(
        metric_sample["prediction"],
        metric_sample["target"],
        n_bins=15,
        num_classes=num_classes,
        task="multiclass",
    )


def accuracy(output, batch, create_metric_sample=create_metric_sample_next_token_task):
    num_classes = output["logits"].shape[-1]
    metric_sample = create_metric_sample(output, batch)
    return tm.functional.accuracy(
        metric_sample["prediction"],
        metric_sample["target"],
        task="multiclass",
        num_classes=num_classes,
    )


def get_OOD_metrics(softmax_probs_CIFAR10, softmax_probs_CIFAR100, softmax_probs_SVHN, args):
    """

    EVALUATE THE PERFORMANCE: OOD CIFAR10, CIFAR100, SVHN
    Input: softmax_probs_CIFAR10, softmax_probs_CIFAR100, softmax_max_probs_SVHN, args
    Output:

    if args.cifar_mode == "CIFAR10":
        {
            "CIFAR10_CIFAR100":result_CIFAR10_CIFAR100,
            "CIFAR10_SVHN":result_CIFAR10_SVHN,
        }
    else:
        {
            "CIFAR100_CIFAR10":result_CIFAR100_CIFAR10,
            "CIFAR100_SVHN":result_CIFAR100_SVHN,
        }

    """

    softmax_probs_CIFAR100 = torch.mean(softmax_probs_CIFAR100, dim=0, keepdim=False)
    softmax_probs_CIFAR10 = torch.mean(softmax_probs_CIFAR10, dim=0, keepdim=False)
    softmax_probs_SVHN = torch.mean(softmax_probs_SVHN, dim=0, keepdim=False)

    softmax_max_probs_CIFAR100 = softmax_probs_CIFAR100.data.max(1, keepdim=False)[0]
    softmax_max_probs_CIFAR10 = softmax_probs_CIFAR10.data.max(1, keepdim=False)[0]
    softmax_max_probs_SVHN = softmax_probs_SVHN.data.max(1, keepdim=False)[0]

    # AUC_ROC
    if args.cifar_mode == "CIFAR10":
        # CIFAR10 + CIFAR100
        max_probs_0 = softmax_max_probs_CIFAR100.sort(descending=True)[0]
        max_probs_1 = softmax_max_probs_CIFAR10.sort(descending=True)[0]
        sorted_classes = create_sorted_classes(max_probs_0, max_probs_1)
        result_CIFAR10_CIFAR100 = compute_AUC_ROC(sorted_classes)
        print("AUC ROC CIFAR10 + CIFAR100: {:.4f}\n".format(result_CIFAR10_CIFAR100))

        # CIFAR10 + SVHN
        max_probs_0 = softmax_max_probs_SVHN.sort(descending=True)[0]
        max_probs_1 = softmax_max_probs_CIFAR10.sort(descending=True)[0]
        sorted_classes = create_sorted_classes(max_probs_0, max_probs_1)
        result_CIFAR10_SVHN = compute_AUC_ROC(sorted_classes)
        print("AUC ROC CIFAR10 + SVHN: {:.4f}\n".format(result_CIFAR10_SVHN))
        return {
            "CIFAR10_CIFAR100": result_CIFAR10_CIFAR100,
            "CIFAR10_SVHN": result_CIFAR10_SVHN,
        }

    if args.cifar_mode == "CIFAR100":
        # CIFAR100 + CIFAR10
        max_probs_0 = softmax_max_probs_CIFAR10.sort(descending=True)[0]
        max_probs_1 = softmax_max_probs_CIFAR100.sort(descending=True)[0]
        sorted_classes = create_sorted_classes(max_probs_0, max_probs_1)
        result_CIFAR100_CIFAR10 = compute_AUC_ROC(sorted_classes)
        print("AUC ROC CIFAR100 + CIFAR10: {:.4f}\n".format(result_CIFAR100_CIFAR10))

        # CIFAR100 + SVHN
        max_probs_0 = softmax_max_probs_SVHN.sort(descending=True)[0]
        max_probs_1 = softmax_max_probs_CIFAR100.sort(descending=True)[0]
        sorted_classes = create_sorted_classes(max_probs_0, max_probs_1)
        result_CIFAR100_SVHN = compute_AUC_ROC(sorted_classes)
        print("AUC ROC CIFAR100 + SVHN: {:.4f}\n".format(result_CIFAR100_SVHN))
        return {
            "CIFAR100_CIFAR10": result_CIFAR100_CIFAR10,
            "CIFAR100_SVHN": result_CIFAR100_SVHN,
        }


# AUC_ROC
def compute_AUC_ROC(sorted_classes):

    # total number of class 1 and class 0
    N = sorted_classes[sorted_classes == True].shape[0]
    M = sorted_classes.shape[0] - N

    # initialize
    num_class_1 = 0
    area = 0

    # loop over the classes
    for i in range(N + M):
        if sorted_classes[i] == True:
            num_class_1 += 1
        else:
            area += num_class_1

    return area / (M * N)
