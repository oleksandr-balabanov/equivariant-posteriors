import sys
sys.path.append('/cephyr/users/olebal/Alvis/lora ensembles/equivariant-posteriors/')
from lib.datasets.commonsense_qa import DataCommonsenseQaConfig, DataCommonsenseQa

MISTRAL_CHECKPOINT = "mistralai/Mistral-7B-v0.1"
def main():

    print("Started")
    
    config = DataCommonsenseQaConfig(dataset_split = "validation", model_checkpoint = MISTRAL_CHECKPOINT)
    dataset = DataCommonsenseQa(config)
    print("Size of valid dataset: ", len(dataset))
    config = DataCommonsenseQaConfig(dataset_split = "train", model_checkpoint = MISTRAL_CHECKPOINT)
    dataset = DataCommonsenseQa(config)
    print("Size of train dataset: ", len(dataset))
    print("Finished")

if __name__ == "__main__":
    main()