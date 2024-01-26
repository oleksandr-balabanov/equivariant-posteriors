from experiments.lora_ensembles.datasets.dataset_config import create_dataset_config_factory

def create_eval_dataset_config(checkpoint:str, eval_dataset:str, max_len_eval:str):

    create_data_config = create_dataset_config_factory(eval_dataset)
    eval_dataset_config = create_data_config(
        checkpoint=checkpoint,
        max_len_val=max_len_eval,
        max_len_train = 1,
        dataset_split="validation",
    )
    return eval_dataset_config