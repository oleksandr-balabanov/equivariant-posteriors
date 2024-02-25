from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import transformers
import torch
import torch.nn.functional as F
from typing import List

from typing import Dict
from lib.train_dataclasses import TrainEpochState
from lib.dataspec import DataSpec
from lib.metric import MetricSample
import lib.serialize_human

TEST_PROMPTS = [
    "The color of the sky is blue.",
    "The square root of 123456789 is 11111.1110606.",
    "Multimodal language model is a type of artificial intelligence that processes and understands information from multiple modalities, such as text, images, and possibly sound, enabling more comprehensive and nuanced interactions and analyses."
]

TRAIN_PROMPTS = [
    "The color of the sky is blue.",
    "The square root of 123456789 is 11111.1110606.",
    "Multimodal language model is a type of artificial intelligence that processes and understands information from multiple modalities, such as text, images, and possibly sound, enabling more comprehensive and nuanced interactions and analyses."
]

@dataclass
class DataCustomLanguageDatasetConfig:
    dataset: str = "custom_language_dataset"
    model_checkpoint: str = "meta-llama/Llama-2-7b-hf"
    max_len: int = 1024
    num_samples: int = None
    dataset_split: str = "train"

    train_prompts:List[str] = field(default_factory=lambda: [TRAIN_PROMPTS])
    test_prompts:List[str] = field(default_factory=lambda: [TEST_PROMPTS])

    def serialize_human(self):
        return lib.serialize_human.serialize_human(self.__dict__)


class DataCustomLanguageDataset(Dataset):
    def __init__(self, data_config: DataCustomLanguageDatasetConfig):
        # data config
        self.data_config = data_config

        # Concatenate all subsets into a single dataset
        if data_config.dataset_split=='train':
            self.dataset = StringDataset(data_config.train_prompts)
        else:
            self.dataset = StringDataset(data_config.test_prompts)

        # If num_samples is specified and is a positive integer, slice the dataset
        if (
            data_config.num_samples
            and isinstance(data_config.num_samples, int)
            and data_config.num_samples > 0
        ):
            self.dataset = self.dataset.select(range(data_config.num_samples))

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            data_config.model_checkpoint, 
            add_prefix_space=True,
            padding='max_length',  
            truncation=True,       
            max_length=data_config.max_len 
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize the dataset
        col_to_delete = ['string']
        self.tokenized_dataset = self.dataset.map(
            self._preprocess, batched=True, remove_columns=col_to_delete
        )

        self.tokenized_dataset.set_format("torch")
        self.collate_fn = transformers.DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.max_token_size = self._find_max_input_size(self.tokenized_dataset)

        # Debugging prints
        self._print_debug_info()

    def _print_debug_info(self):
        """Prints debug information."""
        print("One Prompt: ", self.dataset[0])
        print("One Tokenized Prompt: ", next(iter(self.tokenized_dataset)))
        print("Max one prompt token size: ", self.max_token_size)
        print("Max model token size: ", self.data_config.max_len)
        print("Dataset contains: ", len(self.dataset))

    def _find_max_input_size(self, tokenized_dataset, attention_mask_column='attention_mask'):
        max_size = 0
        for row in tokenized_dataset:
            attention_mask = torch.tensor(row[attention_mask_column]) if not isinstance(row[attention_mask_column], torch.Tensor) else row[attention_mask_column]
            one_indices = (attention_mask == 1).nonzero(as_tuple=True)[0]
            size = self.data_config.max_len-one_indices[0].item()
            max_size = max(max_size, size)
        return max_size

    def _preprocess(self, batch):
        # Extract the text to be tokenized.
        texts = batch["string"]

        # Tokenize the text
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=self.data_config.max_len,
            padding="max_length",
        )

    @staticmethod
    def data_spec(config: DataCustomLanguageDatasetConfig):
        return DataSpec(
            input_shape=torch.Size([1]),
            output_shape=torch.Size([1]),
            target_shape=torch.Size([1]),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.tokenized_dataset[idx]
        data["sample_id"] = [idx]
        return data


class StringDataset(Dataset):
    def __init__(self, strings):
        """
        Args:
            strings (list of str): List of strings.
        """
        self.strings = strings

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        string = self.strings[idx]
        sample = {'string': string}

        return sample