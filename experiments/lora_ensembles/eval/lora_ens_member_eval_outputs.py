from typing import List, Dict, Callable, Tuple
import lib.data_factory as data_factory


import torch
import torchmetrics as tm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.lora_ensembles.utils.lora_ens_metrics import create_metric_sample_single_token
from experiments.lora_ensembles.utils.lora_ens_inference import LORAEnsemble
from experiments.lora_ensembles.utils.lora_ens_file_operations import save_to_dill, load_from_dill
from experiments.lora_ensembles.eval.lora_ens_member_eval_config_dataclass import LoraEnsMemberEvalConfig

from experiments.lora_ensembles.datasets.dataset_utils import create_eval_dataset_config

IGNORE_INDEX = -100

def calculate_softmax_probs(
    output: torch.Tensor, 
    batch: Dict[str, torch.Tensor], 
    metric_sample_creator: Callable = create_metric_sample_single_token,
    eval_tokens:List[int] = None
) -> torch.Tensor:

    metric_sample = metric_sample_creator(output, batch)
    predictions = metric_sample["predictions"]
    if eval_tokens:
        rescaled_predictions = rescale_softmax_probs(predictions, eval_tokens)

    return rescaled_predictions

def rescale_softmax_probs(
    softmax_probs: torch.Tensor, eval_tokens:List[int]
) -> torch.Tensor:
    
    rescaled_softmax_probs = softmax_probs[:, eval_tokens]
    sum_along_last_dim = torch.sum(rescaled_softmax_probs, dim=-1, keepdim=True)
    rescaled_softmax_probs /= sum_along_last_dim

    return rescaled_softmax_probs


def calculate_targets(
    output: torch.Tensor, 
    batch: Dict[str, torch.Tensor], 
    metric_sample_creator: Callable = create_metric_sample_single_token,
    eval_tokens:List[int] = None
) -> torch.Tensor:
    
    metric_sample = metric_sample_creator(output, batch)
    targets = metric_sample["targets"]
    if eval_tokens:
        eval_tokens_tensor = torch.tensor(eval_tokens)
        transformed_targets = torch.empty_like(targets)
        for idx, token in enumerate(eval_tokens_tensor):
            transformed_targets[targets == token] = idx
        targets = transformed_targets

    return targets 

def calculate_member_softmax_probs_and_targets(
    eval_dataset_config:LoraEnsMemberEvalConfig, 
    eval_batch_size:int,
    member_id:int,
    lora_ens: LORAEnsemble, 
    device: torch.device,
    eval_tokens:List[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    lora_ens.load_member(member_id)
    lora_ens.model.train()    
    eval_dataset = data_factory.get_factory().create(eval_dataset_config)
    eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size)
    accumulated_targets = []
    accumulated_probs = []

    with torch.no_grad():
        for i_batch, batch in enumerate(eval_loader):
            if i_batch % 100 == 0:
                print(f"Member {member_id}, Batch: {i_batch}")
            reshaped_batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            try:
                output = lora_ens.member_forward(batch=reshaped_batch)
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                continue
             
            softmax_probs = calculate_softmax_probs(output = output, batch = reshaped_batch, eval_tokens=eval_tokens)
            targets = calculate_targets(output= output, batch= reshaped_batch, eval_tokens=eval_tokens)

            accumulated_targets.append(targets)
            accumulated_probs.append(softmax_probs)

    final_targets = torch.cat(accumulated_targets).detach().cpu()
    final_softmax_probs = torch.cat(accumulated_probs, dim=0).detach().cpu()

    return final_softmax_probs, final_targets

def does_eval_data_exist(
        member_file_path:str, 
        eval_config:LoraEnsMemberEvalConfig
    ):
    try:
        res_dic = load_from_dill(member_file_path)
        loaded_softmax_probs = res_dic["softmax_probs"]
        loaded_targets = res_dic["targets"]
        loaded_eval_config = res_dic["eval_config"]
        if eval_config!=loaded_eval_config:
            print("The data could be loaded but with not matching eval config. Recomputing the output.")
            return False

        return True   
    except:
        print("The data could not be loaded.")
        return False
    
def save_member_eval_data_to_file(
        softmax_probs:torch.Tensor, 
        targets:torch.Tensor, 
        eval_config:LoraEnsMemberEvalConfig,
        save_member_eval_file_path:str,
        ):
    
        res_dic = {}
        res_dic["softmax_probs"] = softmax_probs
        res_dic["targets"] = targets
        res_dic["eval_config"] = eval_config

        save_to_dill(res_dic, save_member_eval_file_path)


def calculate_and_save_ens_softmax_probs_and_targets(
    lora_ens:LORAEnsemble,
    device: torch.device,
    eval_config:LoraEnsMemberEvalConfig,
    save_member_eval_file_path = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
      
    if not does_eval_data_exist(save_member_eval_file_path, eval_config):
        eval_dataset_config = create_eval_dataset_config(
            checkpoint = eval_config.lora_ens_train_config.checkpoint, 
            eval_dataset = eval_config.eval_dataset, 
            max_len_eval = eval_config.max_len_eval
        )
    
        softmax_probs, targets = calculate_member_softmax_probs_and_targets(
                eval_dataset_config=eval_dataset_config, 
                eval_batch_size=eval_config.eval_batch_size,
                member_id = eval_config.member_id,
                lora_ens = lora_ens, 
                device = device,
                eval_tokens = eval_config.eval_tokens
        )
        save_member_eval_data_to_file(softmax_probs, targets, eval_config, save_member_eval_file_path)
