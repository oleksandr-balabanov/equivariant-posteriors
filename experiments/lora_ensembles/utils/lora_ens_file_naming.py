import os
from typing import Union
from experiments.lora_ensembles.eval.lora_ens_evaluate_config_dataclass import LoraEnsEvalConfig
from experiments.lora_ensembles.plot.lora_ens_plot_config_dataclass import LoraEnsPlotConfig

def create_results_dir(lora_ens_eval_config:LoraEnsEvalConfig):

    # train config
    lora_ens_train_config = lora_ens_eval_config.lora_ens_train_config
    
    dir1_name = lora_ens_eval_config.eval_dir_name
    dir2_name = "train_"+lora_ens_train_config.train_dataset
    dir3_name = "ens_"+str(lora_ens_eval_config.n_members)
    dir4_name = "eval1_"+lora_ens_eval_config.eval_dataset_1
    dir5_name = "eval2_"+lora_ens_eval_config.eval_dataset_2

    dir_path = os.path.join(dir1_name, dir2_name, dir3_name, dir4_name+"_"+dir5_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def create_results_dir_per_epoch(dir_path_without_epoch, epoch):

    dir_path = os.path.join(dir_path_without_epoch, str(epoch))
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def create_results_dir_per_epoch_and_dataset_num(dir_path_with_epoch, dataset_num):

    dir_path = os.path.join(dir_path_with_epoch, f"dataset_{str(dataset_num)}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def extract_model_name_from_checkpoint(checkpoint): 
    last_slash_index = checkpoint.rfind('/')
    if last_slash_index == -1:
        model_name = checkpoint
    else:
        model_name = checkpoint[last_slash_index + 1:]
    return model_name



def create_save_metrics_file_name(lora_ens_eval_config:Union[LoraEnsEvalConfig, LoraEnsPlotConfig]):

    # train config
    lora_ens_train_config = lora_ens_eval_config.lora_ens_train_config
    
    # file name
    file_name = "metrics_"
    file_name += extract_model_name_from_checkpoint(lora_ens_train_config.checkpoint)
    file_name += f"_n_{lora_ens_eval_config.n_members}"
    file_name += f"_lr_{lora_ens_train_config.lora_rank}"
    file_name += f"_dr_{lora_ens_train_config.lora_dropout}"
    file_name += f"_rl2_{lora_ens_train_config.regular_l2}"
    file_name += f"_rl2_{lora_ens_train_config.lora_l2}"

    # Replacing all '.' with 'd' and add extension ".dill"
    file_name = file_name.replace(".", "d")+".dill"

    return file_name

def create_save_ens_probs_and_targets_file_name(lora_ens_eval_config:LoraEnsEvalConfig):

    # train config
    lora_ens_train_config = lora_ens_eval_config.lora_ens_train_config
    
    # file name
    file_name = "ens_probs_and_targets_"
    file_name += extract_model_name_from_checkpoint(lora_ens_train_config.checkpoint)
    file_name += f"_n_{lora_ens_eval_config.n_members}"
    file_name += f"_lr_{lora_ens_train_config.lora_rank}"
    file_name += f"_dr_{lora_ens_train_config.lora_dropout}"
    file_name += f"_rl2_{lora_ens_train_config.regular_l2}"
    file_name += f"_rl2_{lora_ens_train_config.lora_l2}"

    # Replacing all '.' with 'd' and add extension ".dill"
    file_name = file_name.replace(".", "d")+".dill"

    return file_name


def create_save_metric_image_file_name(lora_ens_plot_config:LoraEnsPlotConfig, metric_name):

    lora_ens_train_config = lora_ens_plot_config.lora_ens_train_config
    file_name = metric_name + "_" + lora_ens_plot_config.img_save_file_name
    file_name += extract_model_name_from_checkpoint(lora_ens_train_config.checkpoint)
    file_name += f"_n_{lora_ens_plot_config.n_members}"
    file_name += f"_lr_{lora_ens_train_config.lora_rank}"
    file_name += f"_eps_{lora_ens_plot_config.min_train_epochs}_{lora_ens_plot_config.max_train_epochs}"
    
    # Replacing all '.' with 'd' and add extension ".dill"
    file_name = file_name.replace(".", "d")+".png"
    
    return file_name




