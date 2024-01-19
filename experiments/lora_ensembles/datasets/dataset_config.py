from lib.data_registry import DataMMLUConfig, DataCommonsenseQaConfig
from experiments.lora_ensembles.datasets.dataset_consts import MMLU_SS_SUBSETS, MMLU_STEM_SUBSETS


def commonsense_qa_config(checkpoint, max_len_train, max_len_val, dataset_split):
    return DataCommonsenseQaConfig(
        dataset="commonsense_qa",
        model_checkpoint=checkpoint,
        max_len=max_len_train if dataset_split == "train" else max_len_val,
        dataset_split=dataset_split,
    )

def mmlu_config(checkpoint, max_len_train, max_len_val, dataset_split, subset_names):
    train_dataset_split = ["dev", "val"]
    val_dataset_split = ["test"]
    return DataMMLUConfig(
        dataset="cais/mmlu",
        model_checkpoint=checkpoint,
        max_len=max_len_train if dataset_split == "train" else max_len_val,
        dataset_split=train_dataset_split if dataset_split == "train" else val_dataset_split,
        subset_names=subset_names
    )

def create_dataset_config_factory(dataset: str):
    config_funcs = {
        "commonsense_qa": commonsense_qa_config,
        "mmlu_ss": lambda checkpoint, max_len_train, max_len_val, dataset_split: mmlu_config(
            checkpoint, max_len_train, max_len_val, dataset_split, subset_names=MMLU_SS_SUBSETS
        ),
        "mmlu_step": lambda checkpoint, max_len_train, max_len_val, dataset_split: mmlu_config(
            checkpoint, max_len_train, max_len_val, dataset_split, subset_names=MMLU_STEM_SUBSETS
        ),
    }
    config_func = config_funcs.get(dataset.lower())
    if config_func:
        return config_func
    else:
        raise ValueError(f"Unsupported dataset type: {dataset}")
