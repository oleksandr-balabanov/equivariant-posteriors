import torch
import transformers

# Mistral Checkpoint
MISTRAL_CHECKPOINT = "mistralai/Mistral-7B-v0.1"

def main():
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=MISTRAL_CHECKPOINT
        )

if __name__ == "__main__":
    main()