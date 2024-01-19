from typing import Union
from experiments.lora_ensembles.datasets.dataset_config import create_dataset_config_factory
from experiments.lora_ensembles.eval.lora_ens_evaluate_config_dataclass import LoraEnsEvalConfig
from experiments.lora_ensembles.plot.lora_ens_plot_config_dataclass import LoraEnsPlotConfig

def create_eval_dataset_config(lora_ens_eval_config:Union[LoraEnsEvalConfig, LoraEnsPlotConfig], eval_dataset:str):

    create_data_config = create_dataset_config_factory(eval_dataset)
    eval_dataset_config = create_data_config(
        checkpoint=lora_ens_eval_config.lora_ens_train_config.checkpoint,
        max_len_val=lora_ens_eval_config.max_len_eval,
        max_len_train = 1,
        dataset_split="validation",
    )
    return eval_dataset_config