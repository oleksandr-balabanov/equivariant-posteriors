import torch
import gc
import os

from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config

from experiments.lora_ensembles.eval.lora_ens_member_eval_config_dataclass import LoraEnsMemberEvalConfig

from experiments.lora_ensembles.eval.lora_ens_member_eval_config import (
    create_lora_ens_inference_config_factory,
)
from experiments.lora_ensembles.utils.lora_ens_inference import (
    create_lora_ensemble
)
from experiments.lora_ensembles.eval.lora_ens_member_eval_outputs import ( 
    calculate_and_save_ens_softmax_probs_and_targets
)
from experiments.lora_ensembles.utils.lora_ens_file_naming import (
    create_results_dir,
    create_results_dir_per_epoch,
    create_results_dir_per_epoch_and_dataset,
    create_results_dir_per_epoch_dataset_num_and_member_id,
    create_save_probs_and_targets_file_name,  
)
import argparse

def evaluate_lora_one_ens_member(lora_ens_member_eval_config:LoraEnsMemberEvalConfig):

    device = ddp_setup()

    # configure inference config function  
    create_inference_config = create_lora_ens_inference_config_factory(
        lora_ens_member_eval_config.lora_ens_train_config
    )
    # ensemble config
    ensemble_config = create_ensemble_config(
        create_member_config = create_inference_config,
        n_members = lora_ens_member_eval_config.member_id + 1
    )
    
    lora_ensemble = create_lora_ensemble(
        ensemble_config.members, 
        device, 
        checkpoint_epochs = lora_ens_member_eval_config.epoch
    )
    save_results_dir=create_results_dir(lora_ens_member_eval_config)
    save_results_dir=create_results_dir_per_epoch(save_results_dir, lora_ens_member_eval_config.epoch)
    save_results_dir=create_results_dir_per_epoch_and_dataset(save_results_dir, lora_ens_member_eval_config.eval_dataset)
    save_results_dir=create_results_dir_per_epoch_dataset_num_and_member_id(save_results_dir, lora_ens_member_eval_config.member_id)
    save_results_filename=create_save_probs_and_targets_file_name(lora_ens_member_eval_config.lora_ens_train_config)
    save_results_path=os.path.join(save_results_dir, save_results_filename)

    try:
        calculate_and_save_ens_softmax_probs_and_targets(
            lora_ens=lora_ensemble,
            device=device,
            eval_config=lora_ens_member_eval_config,
            save_member_eval_file_path=save_results_path
        ) 
    except Exception as e:
        print(f"An error occurred in main: {e}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--member_id', type=int, required=True)
    parser.add_argument('--eval_dataset', type=str, required=True)
    parser.add_argument('--train_dataset', type=str, required=True)
    args = parser.parse_args()

    # Now you can use args.epoch and args.member_id in your script
    epoch = args.epoch
    member_id = args.member_id
    eval_dataset = args.eval_dataset
    train_dataset = args.train_dataset


    lora_ens_eval_config = LoraEnsMemberEvalConfig(epoch = epoch, member_id = member_id, eval_dataset = eval_dataset)
    lora_ens_eval_config.lora_ens_train_config.train_dataset = train_dataset
    evaluate_lora_one_ens_member(
        lora_ens_member_eval_config=lora_ens_eval_config
    )
    
if __name__ == "__main__":
    main()



