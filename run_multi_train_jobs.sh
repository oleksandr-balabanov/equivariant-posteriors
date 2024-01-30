#!/bin/bash
# Define the number of epochs and members
epochs=15 # Example: 10 epochs
members=4 # Example: 5 members
train_dataset="commonsense_qa"
batch_size=8
lora_l2=0.1
regular_l2=0
lora_dropout=0
learning_rate=5e-06
use_generative_next_token_loss="true"
max_len_train=128
max_len_val=128
current_dir=$(pwd)

source  $current_dir/experiments/lora_ensembles/hf_env.sh
for member_id in $(seq 0 $members); do
   ./srun_single.sh ./run.sh  $current_dir/experiments/lora_ensembles/run_lora_ens_train.py --batch_size $batch_size --epochs $epochs --member_id $member_id --train_dataset $train_dataset --learning_rate $learning_rate --lora_l2 $lora_l2 --regular_l2 $regular_l2 --lora_dropout $lora_dropout --use_generative_next_token_loss $use_generative_next_token_loss --max_len_train $max_len_train --max_len_val $max_len_val
done
