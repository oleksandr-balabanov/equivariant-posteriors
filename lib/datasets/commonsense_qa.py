from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset
from dataclasses import dataclass
import transformers
import torch
import torch.nn.functional as F

from typing import Dict
from lib.train_dataclasses import TrainEpochState
from lib.dataspec import DataSpec
from lib.metric import MetricSample
import lib.serialize_human


@dataclass
class DataCommonsenseQaConfig:
    dataset: str = "commonsense_qa"
    model_checkpoint: str = "meta-llama/Llama-2-7b-hf"
    max_len: int = 1024
    dataset_split: str = "train"
    num_samples: int = None

    def serialize_human(self):
        return lib.serialize_human.serialize_human(self.__dict__)


class DataCommonsenseQa(Dataset):
    def __init__(self, data_config: DataCommonsenseQaConfig):
        # data config
        self.data_config = data_config

        # Load the dataset
        self.dataset = load_dataset(data_config.dataset)[data_config.dataset_split]

        # If num_samples is specified and is a positive integer, slice the dataset
        if (
            data_config.num_samples
            and isinstance(data_config.num_samples, int)
            and data_config.num_samples > 0
        ):
            self.dataset = self.dataset.select(range(data_config.num_samples))


        # Apply the formatting to the dataset
        formatted_dataset = self.dataset.map(self._format_question_answer)

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
        col_to_delete = [
            "id",
            "question",
            "question_concept",
            "choices",
            "formatted_question_answer",
            "answerKey",
        ]
        self.tokenized_dataset = formatted_dataset.map(
            self._preprocess, batched=True, remove_columns=col_to_delete
        )

        self.tokenized_dataset.set_format("torch")
        self.collate_fn = transformers.DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.max_token_size = self._find_max_input_size(self.tokenized_dataset)
        print("Max token size of input sample: ", self.max_token_size)
        print("a: ",  self.tokenizer.encode("A: (a)."))
        print("b: ",  self.tokenizer.encode("A: (b)."))
        print("c: ",  self.tokenizer.encode("A: (c)."))
        print("d: ",  self.tokenizer.encode("A: (d)."))
        print("e: ",  self.tokenizer.encode("A: (e)."))
        print("Decode: ",  self.tokenizer.decode([29890, 29872, 29890, 29883, 29874, 29881, 29883, 29872]))
        print(formatted_dataset[0])

    def _find_max_input_size(self, tokenized_dataset, attention_mask_column='attention_mask'):
        max_size = 0
        for row in tokenized_dataset:
            # Convert the attention mask to a tensor if it's not already
            attention_mask = row[attention_mask_column]
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask)

            # Find the index of the first occurrence of 0
            zero_indices = (attention_mask == 0).nonzero(as_tuple=True)[0]
            if len(zero_indices) > 0:
                size = zero_indices[0].item()  # Get the index as a Python int
            else:
                # If 0 is not found, use the full length of the attention mask
                size = len(attention_mask)

            # Update the maximum size if this row's size is greater
            if size > max_size:
                max_size = size

        return max_size


    def _format_question_answer(self, item):
        question = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]

        # Formatting choices
        formatted_choices = "\n".join(
            [f"({label.lower()}) {choices[i]}" for i, label in enumerate(labels)]
        )

        # Constructing a formatted question-answer string
        formatted_question_answer = f"Q: {question}\nAnswer Choices:\n{formatted_choices}\nA: ({answer_key.lower()})."

        # Return a dictionary with the formatted question-answer pair
        return {"formatted_question_answer": formatted_question_answer}

    def _preprocess(self, batch):
        # Extract the text to be tokenized.
        texts = batch["formatted_question_answer"]

        # Tokenize the text
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=self.data_config.max_len,
            padding="max_length",
        )

    @staticmethod
    def data_spec(config: DataCommonsenseQaConfig):
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
