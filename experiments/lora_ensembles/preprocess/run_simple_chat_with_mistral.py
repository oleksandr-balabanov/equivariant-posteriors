import sys
sys.path.append('/cephyr/users/olebal/Alvis/lora_ensembles/equivariant-posteriors')

import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec  # Ensure this import matches your project structure

import torch
import gc
import os

from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config

from experiments.lora_ensembles.eval.lora_ens_member_eval_config_dataclass import LoraEnsMemberEvalConfig

from experiments.lora_ensembles.eval.lora_ens_member_eval_config import (
    create_lora_ens_inference_config_factory,
)
from experiments.lora_ensembles.utils.lora_ens_inference import (
    create_lora_ensemble
)
from experiments.lora_ensembles.eval.lora_ens_member_eval_outputs import ( 
    calculate_and_save_ens_softmax_probs_and_targets
)
from experiments.lora_ensembles.utils.lora_ens_file_naming import (
    create_results_dir,
    create_results_dir_per_epoch,
    create_results_dir_per_epoch_and_dataset,
    create_results_dir_per_epoch_dataset_num_and_member_id,
    create_save_probs_and_targets_file_name,  
)
import argparse
import transformers
import peft


def chat_lora_one_ens_member(lora_ens_member_eval_config:LoraEnsMemberEvalConfig):

    device = ddp_setup()

    # configure inference config function  
    create_inference_config = create_lora_ens_inference_config_factory(
        lora_ens_member_eval_config.lora_ens_train_config
    )
    # ensemble config
    ensemble_config = create_ensemble_config(
        create_member_config = create_inference_config,
        n_members = lora_ens_member_eval_config.member_id + 1
    )
    
    lora_ensemble = create_lora_ensemble(
        ensemble_config.members, 
        device, 
        checkpoint_epochs = lora_ens_member_eval_config.epoch
    )

    lora_ensemble.load_member(0)
    #input_text = "Hi, how are you?"
    #lora_ensemble.generate(input_text, max_length=50)

    tokenizer = transformers.AutoTokenizer.from_pretrained(lora_ensemble.model.config.checkpoint, padding_side='left')
    print("Let's chat! (type 'quit' to exit)")
    full_conversation = ""
    #stop_phrase = '"User"'
    while True:
        user_input = input("Please enter a prompt to complete (or 'q' to quit): ")
        if user_input == "q":
            print("Quitting the program.")
            break
        
        #if user_input!="":
        #    full_conversation = f' "User": "{user_input}"; "Chat Bot Assistant": ' 
        
        #full_conversation = user_input
        print(f"Input Prompt: {user_input}")

        
        inputs = tokenizer.encode(user_input, return_tensors='pt').to(lora_ensemble.model.model.device)
        attention_mask = torch.ones_like(inputs)
        inputs = torch.cat([inputs])  # Adjust for device
        #print(inputs)
        outputs = lora_ensemble.member_generate(input_ids=inputs, attention_mask=attention_mask, max_length=200, num_beams=1, do_sample=False)
        #print(outputs)
        
        # Decoding and printing the model's response
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Check if the stop phrase is in the generated text and truncate if found
        #if stop_phrase in response_text[len(full_conversation):]:
        #    stop_index = response_text[len(full_conversation):].index(stop_phrase)
        #    truncated_text = response_text[:stop_index+len(full_conversation)]
        #else:
        #    truncated_text = response_text
        #conversation_text = truncated_text[len(full_conversation):]
        print("Bot's Completion: ", response_text)
        #full_conversation += f'{conversation_text}' 

       


def main():
    # Parse command-line arguments
    member_id=0 
    eval_dataset="custom_language_dataset"
    train_dataset="custom_language_dataset"
    epoch = int(input("Choose epoch: "))
    print(f"The chosen epoch is {epoch}")




    lora_ens_eval_config = LoraEnsMemberEvalConfig(epoch = epoch, member_id = member_id, eval_dataset = eval_dataset)
    lora_ens_eval_config.lora_ens_train_config.train_dataset = train_dataset
    chat_lora_one_ens_member(
        lora_ens_member_eval_config=lora_ens_eval_config
    )
    
if __name__ == "__main__":
    main()



