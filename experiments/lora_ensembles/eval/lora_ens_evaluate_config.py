from experiments.lora_ensembles.eval.lora_ens_evaluate_config_dataclass import LoraEnsEvalConfig
from experiments.lora_ensembles.train.lora_ens_train_config_dataclass import LoraEnsTrainConfig
from experiments.lora_ensembles.train.lora_ens_train_config import create_lora_ens_train_run_config
from lib.data_registry import DataCommonsenseQaConfig

def create_lora_ens_eval_config():
 
    # default eval config         
    lora_ens_eval_config = LoraEnsEvalConfig()

    return  lora_ens_eval_config

def create_lora_ens_inference_config_factory(ens_train_config: LoraEnsTrainConfig):
    def create_config(ensemble_id: int):
        return create_lora_ens_inference_config(ensemble_id, ens_train_config)
    return create_config


def create_lora_ens_inference_config(ensemble_id: int, ens_train_config: LoraEnsTrainConfig) -> dict:
    config = create_lora_ens_train_run_config(ensemble_id, ens_train_config)
    config.compute_config.distributed = False
    config.compute_config.num_gpus = 1
    return config

