from lib.datasets.sine import DataSineConfig, DataSine
from lib.datasets.spiral import DataSpiralsConfig, DataSpirals
from lib.datasets.uniform import DataUniformConfig, DataUniform
from lib.datasets.mnist import DataMNISTConfig, DataMNIST
from lib.datasets.mnist_2 import DataMNIST2Config, DataMNIST2
from lib.datasets.cifar import DataCIFARConfig, DataCIFAR
from lib.datasets.cifar_c import DataCIFAR10CConfig, DataCIFAR10C
from lib.datasets.cifar_2 import DataCIFAR2Config, DataCIFAR2
from lib.datasets.subset import DataSubsetConfig, DataSubset
from lib.datasets.join import DataJoinConfig, DataJoin
from lib.datasets.commonsense_qa import DataCommonsenseQaConfig, DataCommonsenseQa
from lib.datasets.mmlu import DataMMLUConfig, DataMMLU

# from experiments.lora_ensembles.configs import NLPDatasetConfig
# from experiments.lora_ensembles.configs import NLPDataset


def register_datasets():
    datasets = dict()
    datasets[DataSpiralsConfig.__name__] = DataSpirals
    datasets[DataSineConfig.__name__] = DataSine
    datasets[DataUniformConfig.__name__] = DataUniform
    datasets[DataMNISTConfig.__name__] = DataMNIST
    datasets[DataMNIST2Config.__name__] = DataMNIST2
    datasets[DataCIFARConfig.__name__] = DataCIFAR
    datasets[DataCIFAR10CConfig.__name__] = DataCIFAR10C
    datasets[DataCIFAR2Config.__name__] = DataCIFAR2
    datasets[DataSubsetConfig.__name__] = DataSubset
    datasets[DataJoinConfig.__name__] = DataJoin
    datasets[DataCommonsenseQaConfig.__name__] = DataCommonsenseQa
    datasets[DataMMLUConfig.__name__] = DataMMLU
    return datasets
