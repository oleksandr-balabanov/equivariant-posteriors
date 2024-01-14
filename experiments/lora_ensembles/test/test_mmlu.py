import sys
sys.path.append('/cephyr/users/olebal/Alvis/lora ensembles/equivariant-posteriors/')
from lib.datasets.mmlu import DataMMLUConfig, DataMMLU


import sys
sys.path.append('/cephyr/users/olebal/Alvis/lora ensembles/equivariant-posteriors/')
from lib.datasets.commonsense_qa import DataCommonsenseQaConfig, DataCommonsenseQa

MISTRAL_CHECKPOINT = "mistralai/Mistral-7B-v0.1"
def main():

    print("Started")
    subset_names = ["all"]
    config = DataMMLUConfig(dataset_split = "validation", subset_names = subset_names, model_checkpoint = MISTRAL_CHECKPOINT)
    dataset = DataMMLU(config)
    print("Size of valid dataset: ", len(dataset))
    config = DataMMLUConfig(dataset_split = "train", subset_names = subset_names, model_checkpoint = MISTRAL_CHECKPOINT)
    dataset = DataMMLU(config)
    print("Size of train dataset: ", len(dataset))
    print("Finished")

if __name__ == "__main__":
    main()