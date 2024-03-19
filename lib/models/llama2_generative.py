from dataclasses import dataclass, field
import transformers
import peft
import torch
from torch import Tensor, nn
from lib.dataspec import DataSpec
import lib.serialize_human
from typing import List, Dict
import lib.ddp as ddp
from transformers import AutoTokenizer

@dataclass
class LLaMA2GenerativeConfig:
    """
    Configuration class for LLaMA 2 Generative model.

    Attributes:
    - checkpoint (str): Pretrained model checkpoint path.
    - lora_rank (int): LoRA rank for model adjustments.
    - lora_alpha (float): LoRA alpha value for scaling.
    - lora_dropout (float): Dropout rate for LoRA layers.
    - lora_l2 (float): L2 regularization term for LoRA.
    - target_modules (List[str]): List of module names to target for LoRA adjustments.
    """
    checkpoint: str = "meta-llama/Llama-2-7b-hf"
    lora_rank: int = 8
    lora_alpha: float = 16
    lora_dropout: float = 0.0
    lora_l2: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    def serialize_human(self) -> str:
        """Converts configuration data to a human-readable string."""
        return lib.serialize_human.serialize_human(self.__dict__)

class LLaMA2Generative(nn.Module):
    """
    PyTorch module for LLaMA 2 Generative model with PEFT (Parameter-efficient Fine-tuning).

    Attributes:
    - model_config (LLaMA2GenerativeConfig): Configuration for the LLaMA 2 model.
    - data_config (DataSpec): Data specification for input data.
    """
    def __init__(self, model_config: LLaMA2GenerativeConfig, data_config: DataSpec):
        super().__init__()
        self.config = model_config
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_config.checkpoint,
            device_map=ddp.get_rank(),
        )

        self.peft_config = peft.LoraConfig(
            task_type="CAUSAL_LM",
            r=model_config.lora_rank,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            bias="none",
            target_modules=model_config.target_modules,
        )
        self.model = peft.get_peft_model(self.base_model, self.peft_config)
        self.setup_tokenizer()
        self.device = next(self.model.parameters()).device
        
    def setup_tokenizer(self, max_len=128):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.checkpoint, 
            add_prefix_space=True,
            padding='max_length',  
            truncation=True,       
            max_length=max_len
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Forward pass for the model.

        Args:
        - batch (Dict[str, Tensor]): Input batch containing 'input_ids' and 'attention_mask'.

        Returns:
        - Dict[str, Tensor]: Model outputs including logits and optionally LoRA L2 loss.
        """
        outputs = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        outputs = dict(logits=outputs["logits"])

        outputs["lora_l2_loss"] = torch.tensor(0.0).to(self.device)
        if self.config.lora_l2 > 0:
            outputs["lora_l2_loss"] = self.config.lora_l2 * self.lora_l2_loss()
        return outputs

    def lora_l2_loss(self) -> Tensor:
        """
        Calculate the L2 loss for LoRA parameters.

        Returns:
        - Tensor: The calculated L2 loss.
        """
        lora_l2_loss = torch.tensor(0.0).to(self.device)
        lora_pairs = {}

        for name, param in self.model.named_parameters():
            if "lora" in name:
                last_lora_index = name.rfind("lora")
                base_name = name[:last_lora_index]

                if base_name not in lora_pairs:
                    lora_pairs[base_name] = []
                lora_pairs[base_name].append(param)

        for base_name, matrices in lora_pairs.items():
            if len(matrices) == 2:
                loraA, loraB = matrices
                total_matrix = loraB @ loraA
                lora_l2_loss += torch.norm(total_matrix, 2)**2

        return lora_l2_loss

    def state_dict(self, **kwargs) -> Dict[str, Tensor]:
        """
        Override state_dict with only adapter weights.

        Correct key mismatches in the state_dict due to naming differences in LoRA layers.
        Specifically, this modifies the keys to include the '.default.' segment where necessary,
        aligning the keys in the provided state_dict with the format expected by the PEFT model.

        Args:
        - kwargs: Additional arguments.

        Returns:
        - Dict[str, Tensor]: State dictionary with adapter weights.
        """
        state_dict = peft.get_peft_model_state_dict(self.model)
        updated_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("lora_A.weight", "lora_A.default.weight")
            new_key = new_key.replace("lora_B.weight", "lora_B.default.weight")
            updated_state_dict[new_key] = value
        prefix = ""
        updated_state_dict = {f"{prefix}{k}": v for k, v in updated_state_dict.items()}
        return updated_state_dict

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = False) -> None:
        
        """
        Override state_dict to load adapter weights with non-strict loading.

        Args:
        - state_dict (Dict[str, Tensor]): State dictionary to load.
        - strict (bool): Indicates whether strict loading should be enforced. 
          If set to False, which is the default, the loading will not be strict.
        """
        self.model.load_state_dict(state_dict, strict=strict)
