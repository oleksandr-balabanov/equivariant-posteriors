from experiments.lora_ensembles.plot.lora_ens_plot_config_dataclass import LoraEnsPlotConfig
from experiments.lora_ensembles.eval.lora_ens_evaluate_config_dataclass import LoraEnsEvalConfig

def convert_eval_config_to_plot_config(eval_config: LoraEnsEvalConfig) -> LoraEnsPlotConfig:
    """
    Convert a LoraEnsEvalConfig instance to a LoraEnsPlotConfig instance.
    """
    return LoraEnsPlotConfig(
        n_members=eval_config.n_members,
        min_train_epochs=eval_config.min_train_epochs,
        max_train_epochs=eval_config.max_train_epochs,
        max_len_eval=eval_config.max_len_eval,
        eval_dataset_1_config=eval_config.eval_dataset_1_config,
        eval_dataset_2_config=eval_config.eval_dataset_2_config,
        eval_dir_name=eval_config.eval_dir_name,
        load_softmax_probs=eval_config.load_softmax_probs,
        lora_ens_train_config=eval_config.lora_ens_train_config  
    )

def convert_plot_config_to_eval_config(plot_config: LoraEnsPlotConfig) -> LoraEnsEvalConfig:
    """
    Convert a LoraEnsPlotConfig instance to a LoraEnsEvalConfig instance.
    """
    return LoraEnsEvalConfig(
        n_members=plot_config.n_members,
        min_train_epochs=plot_config.min_train_epochs,
        max_train_epochs=plot_config.max_train_epochs,
        max_len_eval=plot_config.max_len_eval,
        eval_dataset_1_config=plot_config.eval_dataset_1_config,
        eval_dataset_2_config=plot_config.eval_dataset_2_config,
        eval_dir_name=plot_config.eval_dir_name,
        load_softmax_probs=plot_config.load_softmax_probs,
        lora_ens_train_config=plot_config.lora_ens_train_config  
    )
