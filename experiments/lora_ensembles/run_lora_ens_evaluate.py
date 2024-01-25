import torch
import gc
import os

from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config

from experiments.lora_ensembles.eval.lora_ens_evaluate_config_dataclass import LoraEnsEvalConfig

from experiments.lora_ensembles.eval.lora_ens_evaluate_config import (
    create_lora_ens_inference_config_factory,
    create_lora_ens_eval_config
)
from experiments.lora_ensembles.utils.lora_ens_inference import (
    create_lora_ensemble
)
from experiments.lora_ensembles.eval.lora_ens_evaluate_metrics import ( 
    evaluate_lora_ens_on_two_datasets_and_ood, 
    evaluate_two_lora_ens_and_agr_var
)
from experiments.lora_ensembles.datasets.dataset_utils import (
    create_eval_dataset_config
)
from experiments.lora_ensembles.utils.lora_ens_file_naming import (
    create_results_dir,
    create_results_dir_per_epoch,
    create_results_dir_per_epoch_and_dataset,
    create_save_metrics_file_name,
    create_save_probs_and_targets_file_name,  
    create_save_agr_var_file_name
)
from experiments.lora_ensembles.utils.lora_ens_file_operations import (
    save_to_dill
)

def clean_gpu():
    torch.cuda.empty_cache()
    gc.collect()

def evaluate_lora_ens_ood(lora_ens_eval_config:LoraEnsEvalConfig):

    eval_dataset_1_config = create_eval_dataset_config(
        lora_ens_eval_config.lora_ens_train_config_1.checkpoint, 
        lora_ens_eval_config.eval_dataset_1,
        lora_ens_eval_config.max_len_eval_1,
        )
    eval_dataset_2_config = create_eval_dataset_config(
        lora_ens_eval_config.lora_ens_train_config_1.checkpoint,
        lora_ens_eval_config.eval_dataset_2,
        lora_ens_eval_config.max_len_eval_2,
        )

    # configure inference config function  
    create_inference_config = create_lora_ens_inference_config_factory(
        lora_ens_eval_config.lora_ens_train_config_1
    )

    # ensemble config
    ensemble_config = create_ensemble_config(
        create_member_config = create_inference_config,
        n_members = lora_ens_eval_config.n_members_1
    )
    save_results_dir = create_results_dir(lora_ens_eval_config)
    try:
        device = ddp_setup()
        min_train_epochs = lora_ens_eval_config.min_train_epochs
        max_train_epochs = lora_ens_eval_config.max_train_epochs
        for train_epochs in range(min_train_epochs, max_train_epochs + 1):
            
            # create save dir
            save_results_dir_per_epoch = create_results_dir_per_epoch(save_results_dir, train_epochs)

            print("------------------------")
            print(f"Checkpoint Epoch: {train_epochs}/{max_train_epochs}")

            # lora ensemble
            lora_ensemble = create_lora_ensemble(ensemble_config.members, device, checkpoint_epochs = train_epochs)

            
            # save path dir and file
            dir1=None
            dir2=None
            file_name1 = None
            file_name2 = None
            if lora_ens_eval_config.load_softmax_probs:
                eval_dataset_1 = lora_ens_eval_config.eval_dataset_1
                eval_dataset_2 = lora_ens_eval_config.eval_dataset_2
                dir_1 = create_results_dir_per_epoch_and_dataset(save_results_dir_per_epoch, eval_dataset_1)
                dir_2 = create_results_dir_per_epoch_and_dataset(save_results_dir_per_epoch, eval_dataset_2)
                file_name_1 = create_save_probs_and_targets_file_name(lora_ens_eval_config.lora_ens_train_config_1)

            
            ens_result_per_epoch = evaluate_lora_ens_on_two_datasets_and_ood(
                dataset_1_config = eval_dataset_1_config, 
                dataset_2_config = eval_dataset_2_config, 
                eval_batch_size_1 = lora_ens_eval_config.eval_batch_size_1,
                eval_batch_size_2 = lora_ens_eval_config.eval_batch_size_2,
                lora_ensemble = lora_ensemble, 
                device = device,
                save_file_dir_1=dir_1,
                save_file_dir_2=dir_2,            
                save_file_name_1=file_name_1,
                save_file_name_2= file_name_1,
                eval_tokens = lora_ens_eval_config.eval_tokens
            )

            # free gpu
            lora_ensemble = None
            clean_gpu()

            # save res to file
            res = {"ens_result_per_epoch": ens_result_per_epoch}
            file_name = create_save_metrics_file_name(lora_ens_eval_config)
            file_path =  os.path.join(save_results_dir_per_epoch, file_name)
            save_to_dill(res, file_path = file_path)
            print("Result is saved to ", file_path)

    except Exception as e:
        print(f"An error occurred in main: {e}")




def evaluate_lora_ens_agr_var(lora_ens_eval_config:LoraEnsEvalConfig):


    eval_dataset_1_config = create_eval_dataset_config(
        lora_ens_eval_config.lora_ens_train_config_1.checkpoint, 
        lora_ens_eval_config.eval_dataset_1,
        lora_ens_eval_config.max_len_eval_1,
        )

    # configure inference config function  
    create_inference_config_1 = create_lora_ens_inference_config_factory(
        lora_ens_eval_config.lora_ens_train_config_1
    )
    create_inference_config_2 = create_lora_ens_inference_config_factory(
        lora_ens_eval_config.lora_ens_train_config_2
    )

    # ensemble config
    ensemble_config_1 = create_ensemble_config(
        create_member_config = create_inference_config_1,
        n_members = lora_ens_eval_config.n_members_1
    )
    ensemble_config_2 = create_ensemble_config(
        create_member_config = create_inference_config_2,
        n_members = lora_ens_eval_config.n_members_2
    )
    save_results_dir = create_results_dir(lora_ens_eval_config)
    try:
        device = ddp_setup()
        min_train_epochs = lora_ens_eval_config.min_train_epochs
        max_train_epochs = lora_ens_eval_config.max_train_epochs
        for train_epochs in range(min_train_epochs, max_train_epochs + 1):
            
            # create save dir
            save_results_dir_per_epoch = create_results_dir_per_epoch(save_results_dir, train_epochs)

            print("------------------------")
            print(f"Checkpoint Epoch: {train_epochs}/{max_train_epochs}")

            # lora ensemble
            lora_ensemble_1 = create_lora_ensemble(ensemble_config_1.members, device, checkpoint_epochs = train_epochs)
            lora_ensemble_2 = create_lora_ensemble(ensemble_config_2.members, device, checkpoint_epochs = train_epochs)
            
            # save path dir and file
            dir1=None
            file_name1 = None
            if lora_ens_eval_config.load_softmax_probs:
                eval_dataset_1 = lora_ens_eval_config.eval_dataset_1
                dir_1 = create_results_dir_per_epoch_and_dataset(save_results_dir_per_epoch, eval_dataset_1)
                file_name_1 = create_save_probs_and_targets_file_name(lora_ens_eval_config.lora_ens_train_config_1)
                file_name_2 = create_save_probs_and_targets_file_name(lora_ens_eval_config.lora_ens_train_config_2)
  
            
            ens_result_per_epoch = evaluate_two_lora_ens_and_agr_var(
                dataset_config = eval_dataset_1_config, 
                eval_batch_size = lora_ens_eval_config.eval_batch_size_1,
                lora_ensemble_1 = lora_ensemble_1, 
                lora_ensemble_2 = lora_ensemble_2, 
                device = device,
                save_file_dir=dir_1,          
                save_file_name_1=file_name_1,
                save_file_name_2=file_name_2,
                eval_tokens = lora_ens_eval_config.eval_tokens
            )

            # free gpu
            lora_ensemble_1 = None
            lora_ensemble_2 = None
            clean_gpu()

            # save res to file
            res = {"ens_result_per_epoch": ens_result_per_epoch}
            file_name = create_save_agr_var_file_name(lora_ens_eval_config)
            file_path =  os.path.join(save_results_dir_per_epoch, file_name)
            save_to_dill(res, file_path = file_path)
            print("Result is saved to ", file_path)

    except Exception as e:
        print(f"An error occurred in main: {e}")

def main():
    lora_ens_eval_config = create_lora_ens_eval_config()
    if lora_ens_eval_config.eval_ood:
        evaluate_lora_ens_ood(lora_ens_eval_config)

    if lora_ens_eval_config.eval_agr_var:
        evaluate_lora_ens_agr_var(lora_ens_eval_config)

    

if __name__ == "__main__":
    main()



