from experiments.lora_ensembles.plot.lora_ens_plot_config_dataclass import LoraEnsPlotConfig
from lib.data_registry import DataCommonsenseQaConfig


def create_lora_ens_plot_config():

    # default plot config         
    lora_ens_plot_config_default = LoraEnsPlotConfig()

    # in-domain dataset
    in_domain_dataset_config = DataCommonsenseQaConfig(
        dataset="commonsense_qa",
        model_checkpoint=lora_ens_plot_config_default.lora_ens_train_config.checkpoint,
        max_len=lora_ens_plot_config_default.max_len_eval,
        dataset_split="validation",
    )

    # out-of-domain dataset
    out_of_domain_dataset_config = DataCommonsenseQaConfig(
        dataset="commonsense_qa",
        model_checkpoint=lora_ens_plot_config_default.lora_ens_train_config.checkpoint,
        max_len=lora_ens_plot_config_default.max_len_eval,
        dataset_split="validation",
    )
    
    lora_ens_plot_config = LoraEnsPlotConfig(
        eval_dataset_1_config = in_domain_dataset_config,
        eval_dataset_2_config = out_of_domain_dataset_config,        
    )

    return  lora_ens_plot_config