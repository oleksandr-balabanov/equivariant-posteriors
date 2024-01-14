from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset
from dataclasses import dataclass
import transformers
import torch
from typing import Dict
from lib.train_dataclasses import TrainEpochState
from lib.dataspec import DataSpec
from lib.metric import MetricSample
import lib.serialize_human

@dataclass
class DataCommonsenseQaConfig:
    """
    Configuration class for Commonsense QA data.

    Attributes:
    - dataset (str): Name of the dataset.
    - model_checkpoint (str): Pretrained model checkpoint path.
    - max_len (int): Maximum length for tokenization.
    - dataset_split (str): Split of the dataset to use (e.g., 'train', 'test').
    - num_samples (int): Number of samples to use from the dataset; None for using all.
    """
    dataset: str = "commonsense_qa"
    model_checkpoint: str = "meta-llama/Llama-2-7b-hf"
    max_len: int = 1024
    dataset_split: str = "train"
    num_samples: int = None

    def serialize_human(self):
        """Converts configuration data to a human-readable string."""
        return lib.serialize_human.serialize_human(self.__dict__)

class DataCommonsenseQa(Dataset):
    """
    Dataset class for Commonsense QA.

    Methods:
    - __init__: Initializes the dataset.
    - _find_max_input_size: Finds the maximum input size in the tokenized dataset.
    - _format_question_answer: Formats the question and answer for tokenization.
    - _preprocess: Tokenizes the batch of formatted questions and answers.
    - __len__: Returns the length of the dataset.
    - __getitem__: Retrieves an item by its index.
    """
    def __init__(self, data_config: DataCommonsenseQaConfig):
        self.data_config = data_config
        self.dataset = load_dataset(data_config.dataset)[data_config.dataset_split]
        if data_config.num_samples and isinstance(data_config.num_samples, int) and data_config.num_samples > 0:
            self.dataset = self.dataset.select(range(data_config.num_samples))

        formatted_dataset = self.dataset.map(self._format_question_answer)
        self.tokenizer = AutoTokenizer.from_pretrained(
            data_config.model_checkpoint, 
            add_prefix_space=True,
            padding='max_length',  
            truncation=True,       
            max_length=data_config.max_len 
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token

        col_to_delete = [
            "id", "question", "question_concept", "choices",
            "formatted_question_answer", "answerKey",
        ]
        self.tokenized_dataset = formatted_dataset.map(
            self._preprocess, batched=True, remove_columns=col_to_delete
        )
        self.tokenized_dataset.set_format("torch")
        self.collate_fn = transformers.DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.max_token_size = self._find_max_input_size(self.tokenized_dataset)

        # Debugging prints
        self._print_debug_info(formatted_dataset)

    def _print_debug_info(self, formatted_dataset):
        """Prints debug information."""
        print("Max token size of input sample: ", self.max_token_size)
        print("a: ",  self.tokenizer.encode("A: (a)."))
        print("b: ",  self.tokenizer.encode("A: (b)."))
        print("c: ",  self.tokenizer.encode("A: (c)."))
        print("d: ",  self.tokenizer.encode("A: (d)."))
        print("e: ",  self.tokenizer.encode("A: (e)."))
        print(formatted_dataset[0])

    def _find_max_input_size(self, tokenized_dataset, attention_mask_column='attention_mask'):
        max_size = 0
        for row in tokenized_dataset:
            attention_mask = torch.tensor(row[attention_mask_column]) if not isinstance(row[attention_mask_column], torch.Tensor) else row[attention_mask_column]
            zero_indices = (attention_mask == 0).nonzero(as_tuple=True)[0]
            size = zero_indices[0].item() if len(zero_indices) > 0 else len(attention_mask)
            max_size = max(max_size, size)
        return max_size

    def _format_question_answer(self, item):
        question = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]

        formatted_choices = "\n".join([f"({label.lower()}) {choices[i]}" for i, label in enumerate(labels)])
        formatted_question_answer = f"Q: {question}\nAnswer Choices:\n{formatted_choices}\nA: ({answer_key.lower()})."
        return {"formatted_question_answer": formatted_question_answer}

    def _preprocess(self, batch):
        texts = batch["formatted_question_answer"]
        return self.tokenizer(texts, truncation=True, max_length=self.data_config.max_len, padding="max_length")

    @staticmethod
    def data_spec(config: DataCommonsenseQaConfig):
        return DataSpec(input_shape=torch.Size([1]), output_shape=torch.Size([1]), target_shape=torch.Size([1]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.tokenized_dataset[idx]
        data["sample_id"] = [idx]
        return data
