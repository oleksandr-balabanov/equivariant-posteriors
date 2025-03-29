from datasets import Dataset
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from torch.utils.data import Dataset as TorchDataset
from lib.dataspec import DataSpec
from typing import List
import lib
import transformers
import torch
import json

TRAIN_PROMPTS = [
    "The color of the sky is blue.",
    "The square root of 123456789 is 11111.1110606.",
    "Multimodal language model is a type of artificial intelligence that processes and understands information from multiple modalities, such as text, images, and possibly sound, enabling more comprehensive and nuanced interactions and analyses."
]

TEST_PROMPTS = [
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
    
    save_detokenized_text_path:str = "./experiments/lora_ensembles/paper_plots/rebuttal/detokenized_text.json" #None # (optional) path to .json for storing the text represented as tokens
    save_tokenizer_vocabulary_path:str = "./experiments/lora_ensembles/paper_plots/rebuttal/tokenizer_vocabulary.json"#None # (optional) path to .json for storing the full tokenizer vocabulary

    train_prompts: List[str] = field(default_factory=lambda: TRAIN_PROMPTS)
    test_prompts: List[str] = field(default_factory=lambda: TEST_PROMPTS)

    def serialize_human(self):
        # Assuming the implementation is provided elsewhere
        return lib.serialize_human.serialize_human(self.__dict__)

class DataCustomLanguageDataset(TorchDataset):
    def __init__(self, data_config: DataCustomLanguageDatasetConfig):
        self.data_config = data_config

        # Create a Hugging Face dataset from the prompts
        self.prompts = data_config.train_prompts if data_config.dataset_split == 'train' else data_config.test_prompts
        dataset = Dataset.from_dict({"prompts": self.prompts})

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            data_config.model_checkpoint, 
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        

        # Tokenize the dataset
        self.tokenized_dataset = dataset.map(
            self._preprocess, batched=True
        )
        self.dataset=dataset

        self.tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        self.collate_fn = transformers.DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.max_token_size = self._find_max_input_size(self.tokenized_dataset)
        
        self.detokenize_dataset = self.get_detokenize_dataset()
        self.tokenizer_vocabulary = self.get_tokenizer_vocabulary()

        # save
        if self.data_config.save_detokenized_text_path:
            with open(self.data_config.save_detokenized_text_path, "w") as json_file:
                json.dump(self.detokenize_dataset, json_file)

        if self.data_config.save_tokenizer_vocabulary_path:
            with open(self.data_config.save_tokenizer_vocabulary_path, "w") as json_file:
                json.dump(self.tokenizer_vocabulary, json_file)

        # Debugging prints
        self._print_debug_info()

    def _preprocess(self, batch):
        texts = batch["prompts"]
        return self.tokenizer(
            texts,
            truncation=True, 
            max_length=self.data_config.max_len, 
            padding='max_length', 
            return_tensors='pt',
        )

    def _print_debug_info(self):
        print("One Prompt: ", self.prompts[0])
        print("One Tokenized Prompt: ", self.tokenized_dataset[0])
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
    
    @staticmethod
    def data_spec(config: DataCustomLanguageDatasetConfig):
        return DataSpec(input_shape=torch.Size([1]), output_shape=torch.Size([1]), target_shape=torch.Size([1]))

    def get_detokenize_dataset(self):
        detokenized_dataset = {}
        token_index = 0
        for _, example in enumerate(self.tokenized_dataset):
            tokens = example["input_ids"]
            # Detokenize the tokens
            for token in tokens:
                detokenized_token_text = self.tokenizer.decode([token], skip_special_tokens=True)
                detokenized_dataset[str(token_index)]=detokenized_token_text
                token_index+=1
        return detokenized_dataset
    
    def get_tokenizer_vocabulary(self):
        tokenizer_vocabulary ={}
        for token in range(32000):
            detokenized_token_text = self.tokenizer.decode([token], skip_special_tokens=True)
            tokenizer_vocabulary[str(token)] = detokenized_token_text
        return tokenizer_vocabulary

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.tokenized_dataset[idx]
        data["sample_id"] = idx
        return data
