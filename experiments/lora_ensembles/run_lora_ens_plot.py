
import os
from experiments.lora_ensembles.plot.lora_ens_plot_config_dataclass import LoraEnsPlotConfig
from experiments.lora_ensembles.utils.lora_ens_file_naming import (
    create_results_dir,
    create_save_metric_image_file_name
)

from experiments.lora_ensembles.plot.lora_ens_plot_config import (
    create_lora_ens_plot_config
)
from experiments.lora_ensembles.plot.lora_ens_plot_const_configs import (
    PLOT_ENS_PARAMS_ENTIRE, 
    PLOT_ENS_PARAMS_REGULARL2,
    PLOT_ENS_PARAMS_LORAL2,
    PLOT_ENS_PARAMS_LORAL2_DR0D1,
    PLOT_ENS_PARAMS_BEST_N_1,
    PLOT_ENS_PARAMS_RANK_N_1,
    PLOT_ENS_PARAMS_BEST_N_5
)
from experiments.lora_ensembles.plot.lora_ens_plot_utils import (
    load_metrics_from_files,
    plot_and_save,
)

def update_config(config, params):
    for param, value in params.items():
        if hasattr(config, param):
            setattr(config, param, value)
        else:
            print(f"Parameter {param} not found in LoraEnsTrainConfig")

def main():
    """
    Main function to evaluate the LORA ensemble on two datasets and calculate OOD performance.
    """


    multiple_results ={}
    metric_name = "ood_score_entropy"

    # plot config         
    lora_ens_plot_config = create_lora_ens_plot_config()

    # plot params
    plot_ens_params = PLOT_ENS_PARAMS_BEST_N_5
    for plot_ens_name in plot_ens_params.keys():
    
        # load params
        one_ens_params = plot_ens_params[plot_ens_name]
        lora_ens_train_config = lora_ens_plot_config.lora_ens_train_config_1
        update_config(lora_ens_train_config, one_ens_params)
        lora_ens_plot_config.lora_ens_train_config_1 = lora_ens_train_config
        
        # load metric values
        multiple_results[plot_ens_name]  = load_metrics_from_files(lora_ens_plot_config, metric_name)

    # plot and save to file
    save_results_dir = create_results_dir(lora_ens_plot_config)
    file_name = create_save_metric_image_file_name(lora_ens_plot_config, metric_name=metric_name)
    file_path =  os.path.join(save_results_dir, file_name)
    plot_and_save(multiple_results, file_path, metric_name)

if __name__ == "__main__":
    main()