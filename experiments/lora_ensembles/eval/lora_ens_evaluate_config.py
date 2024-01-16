from experiments.lora_ensembles.eval.lora_ens_evaluate_config_dataclass import LoraEnsEvalConfig
from experiments.lora_ensembles.train.lora_ens_train_config_dataclass import LoraEnsTrainConfig
from experiments.lora_ensembles.train.lora_ens_train_config import create_lora_ens_train_run_config
from lib.data_registry import DataCommonsenseQaConfig

def create_lora_ens_eval_config():
 
    # default eval config         
    lora_ens_eval_config_default = LoraEnsEvalConfig()

    # in-domain dataset
    in_domain_dataset_config = DataCommonsenseQaConfig(
        dataset="commonsense_qa",
        model_checkpoint=lora_ens_eval_config_default.lora_ens_train_config.checkpoint,
        max_len=lora_ens_eval_config_default.max_len_eval,
        dataset_split="validation",
    )

    # out-of-domain dataset
    out_of_domain_dataset_config = DataCommonsenseQaConfig(
        dataset="commonsense_qa",
        model_checkpoint=lora_ens_eval_config_default.lora_ens_train_config.checkpoint,
        max_len=lora_ens_eval_config_default.max_len_eval,
        dataset_split="validation",
    )
    
    lora_ens_eval_config = LoraEnsEvalConfig(
        eval_dataset_1_config = in_domain_dataset_config,
        eval_dataset_2_config = out_of_domain_dataset_config,        
    )

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

