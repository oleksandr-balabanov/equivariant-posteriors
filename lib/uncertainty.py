from typing import List
from dataclasses import dataclass
import torch
import pandas as pd
from tqdm import tqdm

from lib.ensemble import Ensemble

# E_y[ H(T | D) - H(T | D, x, y)]
# H(y | D) - E_T[H(y | D, x)]


def mean_entropy(probs: torch.Tensor):
    entropy_given_theta = entropy(probs)
    mean_entropy = torch.mean(entropy_given_theta, dim=-1)
    return mean_entropy


def mutual_information(probs: torch.Tensor):
    mu_H = mean_entropy(probs)
    return predictive_entropy(probs) - mu_H


def predictive_entropy(probs: torch.Tensor):
    assert len(probs.shape) == 3
    ensemble_mean = probs.mean(dim=1)
    return entropy(ensemble_mean)


# H - (H - mu_H) = mu_H


def entropy(probs: torch.Tensor):
    eps = torch.finfo(torch.float32).eps
    entropy = -torch.sum(torch.log(probs + eps) * probs, dim=-1)
    return entropy


def test_entropy():
    # Two distributions over 4 classes
    # First uniform, second uniform over first 2 classes
    data = torch.tensor([[1 / 4, 1 / 4, 1 / 4, 1 / 4], [0.5, 0.5, 0, 0]])
    torch.testing.assert_close(entropy(data), torch.log(torch.tensor([4, 2])))


def test_predictive_entropy():
    # Two members with a uniform mean
    data = torch.tensor([[1 / 2, 1 / 2, 0, 0], [0, 0, 1 / 2, 1 / 2]])
    # Reshape to one batch
    data = data.reshape([1, 2, 4])
    # Uniform mean over 4 classes should have entropy ln(4)
    torch.testing.assert_close(predictive_entropy(data), torch.log(torch.tensor([4])))


def test_mutual_information():
    # Two members with a uniform mean
    data = torch.tensor([[1 / 2, 1 / 2, 0, 0], [0, 0, 1 / 2, 1 / 2]])
    # Reshape to one batch
    data = data.reshape([1, 2, 4])
    # Mutual information between two members is the ensemble mean minus member mean
    torch.testing.assert_close(
        mutual_information(data),
        torch.log(torch.tensor([4])) - torch.log(torch.tensor([2])),
    )


@dataclass
class Uncertainty:
    MI: torch.Tensor
    H: torch.Tensor
    A: torch.Tensor
    mean_pred: torch.Tensor
    sample_ids: torch.Tensor
    sample_id_spec: List[str]
    targets: torch.Tensor


def uncertainty(data_loader: torch.utils.data.DataLoader, ensemble: Ensemble, device):
    sample_ids = []
    MIS = []
    HS = []
    AS = []
    mean_preds = []
    targets = []
    for iteration, batch in enumerate(tqdm(data_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        input = batch["input"]
        target = batch["target"]
        sample_id = batch["sample_id"]
        # if iteration > 10:
        #    break
        probs = torch.zeros(
            [input.shape[0], ensemble.n_members, data_loader.dataset.n_classes]
        )
        # input = input.to(device, non_blocking=True)
        for idx, member in enumerate(ensemble.members):
            output = member(batch)
            probs[:, idx, :] = output["predictions"]

        MI = mutual_information(probs)
        H = predictive_entropy(probs)
        A = mean_entropy(probs)
        sample_ids.append(sample_id)
        mean_preds.append(torch.argmax(torch.mean(probs, dim=1), dim=-1))
        targets.append(target)
        MIS.append(MI)
        HS.append(H)
        AS.append(A)

    return Uncertainty(
        MI=torch.concat(MIS),
        H=torch.concat(HS),
        A=torch.concat(AS),
        sample_ids=torch.concat(sample_ids),
        sample_id_spec=data_loader.dataset.sample_id_spec(data_loader.dataset.config),
        mean_pred=torch.concat(mean_preds),
        targets=torch.concat(targets),
    )


def uq_to_dataframe(uq: Uncertainty) -> pd.DataFrame:
    data = torch.concat(
        [
            uq.MI[:, None].cpu(),
            uq.H[:, None].cpu(),
            uq.sample_ids.cpu(),
            uq.mean_pred[:, None].cpu(),
            uq.targets[:, None].cpu(),
            torch.where(
                uq.targets[:, None].cpu() == uq.mean_pred[:, None].cpu(),
                1.0,
                0.0,
            ),
        ],
        dim=-1,
    )
    return pd.DataFrame(
        columns=["MI", "H"] + uq.sample_id_spec + ["pred", "target", "accuracy"],
        data=data.numpy(),
    )
