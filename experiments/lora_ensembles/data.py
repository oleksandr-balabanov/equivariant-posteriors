from dataclasses import dataclass

import transformers
import datasets

import torch

from lib.dataspec import DataSpec
import lib.serialize_human


@dataclass
class NLPDatasetConfig:
    dataset: str
    model_checkpoint: str
    max_len: int = 512
    validation: bool = False

    def serialize_human(self):
        return lib.serialize_human.serialize_human(self.__dict__)


class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, data_config: NLPDatasetConfig):
        self.config = data_config
        data = datasets.load_dataset(data_config.dataset)["train"]
        data = data.train_test_split(0.9, seed=42)
        if data_config.validation:
            self.dataset = data["test"]
        else:
            self.dataset = data["train"]
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            data_config.model_checkpoint, add_prefix_space=True
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token

        def preprocessing_function(examples):
            return self.tokenizer(
                examples["text"], truncation=True, max_length=data_config.max_len
            )

        col_to_delete = ["id", "keyword", "location", "text"]
        self.tokenized_dataset = self.dataset.map(
            preprocessing_function, batched=True, remove_columns=col_to_delete
        )
        self.tokenized_dataset = self.tokenized_dataset.rename_column("target", "label")
        self.tokenized_dataset.set_format("torch")
        self.collate_fn = transformers.DataCollatorWithPadding(tokenizer=self.tokenizer)

    @staticmethod
    def data_spec(config: NLPDatasetConfig):
        return DataSpec(
            input_shape=torch.Size([1]),
            output_shape=torch.Size([1]),
            target_shape=torch.Size([1]),
        )

    def __getitem__(self, idx):
        data = self.tokenized_dataset[idx]
        data["sample_id"] = [idx]
        return data

    def __len__(self):
        return len(self.tokenized_dataset)
