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


@dataclass
class DataMMLUConfig:
    dataset: str = "lukaemon/mmlu"
    model_checkpoint: str = "meta-llama/Llama-2-7b-hf"
    max_len: int = 256
    dataset_split: str = "train"
    num_samples: int = None
    subset_names: List[str] = field(default_factory=lambda: [
        'high_school_european_history', 
        'business_ethics', 
        'clinical_knowledge', 
        'medical_genetics', 
        'high_school_us_history', 
        'high_school_physics', 
        'high_school_world_history', 
        'virology', 
        'high_school_microeconomics', 
        'econometrics', 
        'college_computer_science', 
        'high_school_biology', 
        'abstract_algebra', 
        'professional_accounting', 
        'philosophy', 
        'professional_medicine', 
        'nutrition', 
        'global_facts', 
        'machine_learning', 
        'security_studies', 
        'public_relations', 
        'professional_psychology', 
        'prehistory', 
        'anatomy', 
        'human_sexuality', 
        'college_medicine', 
        'high_school_government_and_politics', 
        'college_chemistry', 
        'logical_fallacies', 
        'high_school_geography', 
        'elementary_mathematics', 
        'human_aging', 
        'college_mathematics', 
        'high_school_psychology', 
        'formal_logic', 
        'high_school_statistics', 
        'international_law', 
        'high_school_mathematics', 
        'high_school_computer_science', 
        'conceptual_physics', 
        'miscellaneous', 
        'high_school_chemistry', 
        'marketing', 
        'professional_law', 
        'management', 
        'college_physics', 
        'jurisprudence', 
        'world_religions', 
        'sociology', 
        'us_foreign_policy', 
        'high_school_macroeconomics', 
        'computer_security', 
        'moral_scenarios', 
        'moral_disputes', 
        'electrical_engineering', 
        'astronomy', 
        'college_biology'
        ])

    def serialize_human(self):
        return lib.serialize_human.serialize_human(self.__dict__)


class DataMMLU(Dataset):
    def __init__(self, data_config: DataMMLUConfig):
        # data config
        self.data_config = data_config

        # List to store each loaded subset
        subset_datasets = []

        # Iterate over each subset name and load the dataset for that subset
        for subset_name in self.data_config.subset_names: 
            sub_dataset = load_dataset(self.data_config.dataset, subset_name)[data_config.dataset_split]
            subset_datasets.append(sub_dataset)

        # Concatenate all subsets into a single dataset
        self.dataset = concatenate_datasets(subset_datasets)

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
            "input",
            "A",
            "B",
            "C",
            "D",
            "target",
            "formatted_question_answer",
        ]
        self.tokenized_dataset = formatted_dataset.map(
            self._preprocess, batched=True, remove_columns=col_to_delete
        )

        self.tokenized_dataset.set_format("torch")
        self.collate_fn = transformers.DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.max_token_size = self._find_max_input_size(self.tokenized_dataset)
        print("Max token size of question: ", self.max_token_size)

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
        # Construct the question part
        question = f"Q: {item['input']}\nAnswer Choices:\n"
        
        # Append each option
        options = ['A', 'B', 'C', 'D']
        for option in options:
            question += f"({option.lower()}) {item[option]}\n"
        
        # Append the answer
        answer = f"A: ({item['target'].lower()})."
        
        # Return a dictionary with the formatted question-answer pair
        return {"formatted_question_answer": question + answer}

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
    def data_spec(config: DataMMLUConfig):
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
