from dataclasses import dataclass, field
import transformers
import peft
import torch
from lib.dataspec import DataSpec
import lib.serialize_human
from typing import List
import lib.ddp as ddp


# Define LLaMA 2 Configuration
@dataclass
class LLaMA2GenerativeConfig:
    checkpoint: str = "meta-llama/Llama-2-7b-hf"
    lora_rank: int = 16
    lora_alpha: float = 16
    lora_dropout: float = 0.0
    lora_l2: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    def serialize_human(self):
        return lib.serialize_human.serialize_human(self.__dict__)


# Define LLaMA 2 Model with PEFT
class LLaMA2Generative(torch.nn.Module):
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
        self.device = next(self.model.parameters()).device

    def forward(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        outputs = dict(logits=outputs["logits"])

        # Add LoRA L2 loss to the output dic
        outputs["lora_l2_loss"] = torch.tensor(0.0).to(self.device)
        if self.config.lora_l2 > 0:
            outputs["lora_l2_loss"] = self.config.lora_l2 * self.lora_l2_loss()
        return outputs

    def lora_l2_loss(self):
        lora_l2_loss = torch.tensor(0.0).to(self.device)
        lora_pairs = {}

        # Group LoRA tensors by base names
        for name, param in self.model.named_parameters():
            if "lora" in name:
                # Find the last occurrence of 'lora'
                last_lora_index = name.rfind("lora")
                # Extract everything up to that point as the base name
                base_name = name[:last_lora_index]

                if base_name not in lora_pairs:
                    lora_pairs[base_name] = []
                lora_pairs[base_name].append(param)

        # Calculate modified L2 loss for each pair
        for base_name, matrices in lora_pairs.items():
            if len(matrices) == 2:  # Ensure there are exactly two matrices in the pair
                loraA, loraB = matrices

                # Perform SVD decomposition in economic (thin) format
                #U_A, S_A, _ = torch.linalg.svd(loraA, full_matrices=False)
                #_, S_B, V_B = torch.linalg.svd(loraB, full_matrices=False)

                # Convert the vector S into a diagonal matrix
                #S_A_matrix = torch.diag_embed(S_A)
                #S_B_matrix = torch.diag_embed(S_B)

                # Perform matrix multiplication on the last two dimensions
                #total_matrix = S_B_matrix @ V_B.t() @ U_A @ S_A_matrix
                #print(total_matrix.shape)
                total_matrix = loraB @ loraA
                #print(total_matrix.shape)
                lora_l2_loss += torch.norm(total_matrix, 2)**2

        return lora_l2_loss

    def state_dict(self, **kwargs):
        """Override state_dict with only adapter weights"""
        state_dict = peft.get_peft_model_state_dict(self.model)
        updated_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("lora_A.weight", "lora_A.default.weight")
            new_key = new_key.replace("lora_B.weight", "lora_B.default.weight")
            updated_state_dict[new_key] = value
        prefix = ""
        # if "prefix" in kwargs:
        #     prefix = kwargs["prefix"]
        updated_state_dict = {f"{prefix}{k}": v for k, v in updated_state_dict.items()}
        return updated_state_dict

    def load_state_dict(self, state_dict, strict=False):
        """Correct key mismatches in the state_dict due to naming differences in LoRA layers.
        Specifically, this modifies the keys to include the '.default.' segment where necessary,
        aligning the keys in the provided state_dict with the format expected by the PEFT model.
        """
        self.model.load_state_dict(state_dict, strict=False)
