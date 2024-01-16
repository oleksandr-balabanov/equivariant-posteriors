
import os
from experiments.lora_ensembles.utils.lora_ens_file_naming import (
    create_results_dir,
    create_save_metric_image_file_name
)

from experiments.lora_ensembles.plot.lora_ens_plot_config import (
    create_lora_ens_plot_config
)
from experiments.lora_ensembles.plot.lora_ens_plot_utils import (
    load_metrics_from_files,
    plot_and_save,
)

def main():
    """
    Main function to evaluate the LORA ensemble on two datasets and calculate OOD performance.
    """

    multiple_results ={}
    metric_name = "acc_one"

    # plot config         
    lora_ens_plot_config = create_lora_ens_plot_config()

    # load metric values
    multiple_results["lora ens 1"]  = load_metrics_from_files(lora_ens_plot_config, metric_name)

    # plot and save to file
    save_results_dir = create_results_dir(lora_ens_plot_config)
    file_name = create_save_metric_image_file_name(lora_ens_plot_config, metric_name=metric_name)
    file_path =  os.path.join(save_results_dir, file_name)
    plot_and_save(multiple_results, file_path, metric_name)

if __name__ == "__main__":
    main()