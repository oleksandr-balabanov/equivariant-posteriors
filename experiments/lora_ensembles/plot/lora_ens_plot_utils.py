
import os
import matplotlib.pyplot as plt
from experiments.lora_ensembles.utils.lora_ens_file_naming import (
    create_results_dir,
    create_results_dir_per_epoch,
    create_save_metrics_file_name
)
from experiments.lora_ensembles.utils.lora_ens_file_operations import (
    save_to_dill, load_from_dill
)
from experiments.lora_ensembles.plot.lora_ens_plot_config_dataclass import LoraEnsPlotConfig

def plot_and_save(multiple_results, file_path, metric_name):
    plt.figure()
    for ens_name in multiple_results.keys():
        ens_res = multiple_results[ens_name]
        x_axis = ens_res["epoch_values"]
        y_axis = ens_res["metric_values"]

        plt.plot(x_axis, y_axis, label=ens_name)
    
    plt.xlabel("Epoch Values")
    plt.xlabel("Metric Values")
    plt.legend()

    plt.title(metric_name)
    plt.savefig(file_path)
    plt.close()

def flatten_result_dic(res_dict):
    """
    Flatten a nested dictionary by removing a specified nested dictionary
    and incorporating its keys into the main dictionary.

    :param nested_dict: The nested dictionary to flatten.
    :return: A flattened dictionary.
    """
    # Extract the nested 'ood_score' dictionary
    ood_score_dict = res_dict.pop('ood_score', {})

    # Merge the 'ood_score' dictionary with the main dictionary
    flattened_dict = {**res_dict, **ood_score_dict}

    return flattened_dict

def load_metrics_from_files(lora_ens_plot_config:LoraEnsPlotConfig, metric_name:str):
    
    save_results_dir = create_results_dir(lora_ens_plot_config)
    metric_values = []
    for train_epochs in range(lora_ens_plot_config.min_train_epochs, lora_ens_plot_config.max_train_epochs+1):
            
        # create save dir
        save_results_dir_per_epoch = create_results_dir_per_epoch(save_results_dir, train_epochs)

        # save to file
        file_name = create_save_metrics_file_name(lora_ens_plot_config)
        file_path =  os.path.join(save_results_dir_per_epoch, file_name)
        eval_res = load_from_dill(file_path = file_path)
        ens_result_per_epoch = eval_res["ens_result_per_epoch"]
        ens_result_per_epoch = flatten_result_dic(ens_result_per_epoch)
        metric_values.append(ens_result_per_epoch[metric_name])
        
    return {
        "metric_name": metric_name,
        "metric_values": metric_values,
        "epoch_values": list(range(lora_ens_plot_config.min_train_epochs, lora_ens_plot_config.max_train_epochs+1))
    }
 





