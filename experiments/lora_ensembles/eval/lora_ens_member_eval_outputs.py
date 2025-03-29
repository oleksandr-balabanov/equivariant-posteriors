from typing import List, Dict, Callable, Tuple
import lib.data_factory as data_factory

import torch
import torchmetrics as tm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.lora_ensembles.utils.lora_ens_metrics import (
    create_metric_sample_single_token,
    create_metric_sample_next_token,
)
from experiments.lora_ensembles.utils.lora_ens_inference import LORAEnsemble
from experiments.lora_ensembles.utils.lora_ens_file_operations import save_to_dill, load_from_dill
from experiments.lora_ensembles.eval.lora_ens_member_eval_config_dataclass import LoraEnsMemberEvalConfig
from experiments.lora_ensembles.datasets.dataset_utils import create_eval_dataset_config

IGNORE_INDEX = -100

def calculate_softmax_probs(
    output: Dict[str, torch.Tensor], 
    batch: Dict[str, torch.Tensor], 
    metric_sample_creator: Callable = create_metric_sample_single_token,
    eval_tokens:List[int] = None
) -> torch.Tensor:

    metric_sample = metric_sample_creator(output, batch)
    predictions = metric_sample["predictions"]
    if eval_tokens:
        rescaled_predictions = reduce_categories_for_softmax_probs(predictions, eval_tokens)
        return rescaled_predictions
    else:
        return predictions


def rescale_softmax_probs(
    softmax_probs: torch.Tensor, eval_tokens:List[int]
) -> torch.Tensor:
    
    rescaled_softmax_probs = softmax_probs[:, :, eval_tokens]
    sum_along_last_dim = torch.sum(rescaled_softmax_probs, dim=-1, keepdim=True)
    rescaled_softmax_probs /= sum_along_last_dim

    return rescaled_softmax_probs

def reduce_categories_for_softmax_probs(
    softmax_probs: torch.Tensor, eval_tokens:List[int]
) -> torch.Tensor:

    eval_tokens = torch.tensor(eval_tokens, dtype=torch.long)    
    # Move eval_tokens to the same device as softmax_probs
    eval_tokens = eval_tokens.to(device=softmax_probs.device)
    
    reduced_softmax_probs = torch.index_select(softmax_probs, -1, eval_tokens)
    sum_last_dim = 1-reduced_softmax_probs.sum(dim=-1, keepdim=True)
    extended_tensor = torch.cat((reduced_softmax_probs, sum_last_dim), dim=-1)    

    return extended_tensor


def calculate_targets(
    output: Dict[str, torch.Tensor], 
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

def calculate_Lora_L2_loss(
    output: Dict[str, torch.Tensor], 
) -> torch.Tensor:
    return output["lora_l2_loss"]
      

def calculate_member_softmax_probs_targets_and_l2(
    eval_dataset_config:LoraEnsMemberEvalConfig, 
    eval_batch_size:int,
    member_id:int,
    lora_ens: LORAEnsemble, 
    device: torch.device,
    eval_tokens:List[int] = None,
    metric_sample_creator: Callable = create_metric_sample_single_token,
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
            softmax_probs = calculate_softmax_probs(
                output = output, 
                batch = reshaped_batch, 
                eval_tokens=eval_tokens,
                metric_sample_creator = metric_sample_creator,
            )
            targets = calculate_targets(
                output= output, 
                batch= reshaped_batch, 
                eval_tokens=eval_tokens,
                metric_sample_creator = metric_sample_creator,
            )
            print(softmax_probs.shape, targets.shape, output['logits'].shape)
            lora_l2_loss = calculate_Lora_L2_loss(output=output)

            accumulated_targets.append(targets)
            accumulated_probs.append(softmax_probs)

    final_targets = torch.cat(accumulated_targets).detach().cpu()
    final_softmax_probs = torch.cat(accumulated_probs, dim=0).detach().cpu()
    final_lora_l2_loss = lora_l2_loss.detach().cpu()

    return final_softmax_probs, final_targets, final_lora_l2_loss, 

def does_eval_data_exist(
        member_file_path:str, 
        eval_config:LoraEnsMemberEvalConfig
    ):
    try:
        res_dic = load_from_dill(member_file_path)
        loaded_softmax_probs = res_dic["softmax_probs"]
        loaded_targets = res_dic["targets"]
        loaded_lora_l2_loss = res_dic["lora_l2_loss"]
        loaded_eval_config = res_dic["eval_config"]
        if eval_config!=loaded_eval_config:
            print("The data could be loaded, but with a non-matching eval config. Recomputing the output...")
            return False
        print("The data has been successfully loaded. No further computations are necessary.")
        return True   
    except:
        print("The data could not be loaded. Computing the output...")
        return False
    
def save_member_eval_data_to_file(
        softmax_probs:torch.Tensor, 
        targets:torch.Tensor, 
        lora_l2_loss:torch.Tensor, 
        eval_config:LoraEnsMemberEvalConfig,
        save_member_eval_file_path:str,
        ):
    
        res_dic = {}
        res_dic["softmax_probs"] = softmax_probs
        res_dic["targets"] = targets
        res_dic["lora_l2_loss"] = lora_l2_loss
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

        if eval_config.eval_metric_type == "next_token":
            metric_sample_creator=create_metric_sample_next_token
        elif eval_config.eval_metric_type == "single_token":
            metric_sample_creator=create_metric_sample_single_token
        else:
            print(f'eval_config.eval_metric_type is {eval_config.eval_metric_type} that is not supported. Please select from "next token" or "single token" options.')
            raise

        softmax_probs, targets, lora_l2_loss = calculate_member_softmax_probs_targets_and_l2(
                eval_dataset_config=eval_dataset_config, 
                eval_batch_size=eval_config.eval_batch_size,
                member_id = eval_config.member_id,
                lora_ens = lora_ens, 
                device = device,
                eval_tokens = eval_config.eval_tokens,
                metric_sample_creator = metric_sample_creator,
        )
        save_member_eval_data_to_file(softmax_probs, targets, lora_l2_loss, eval_config, save_member_eval_file_path)
