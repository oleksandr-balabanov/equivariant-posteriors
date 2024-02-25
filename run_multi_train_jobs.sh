#!/bin/bash
# Define the number of epochs and members
epochs=0 # Example: 10 epochs
members=0 # Example: 5 members
train_dataset="mmlu_ss"
batch_size=2
lora_l2=1.0
regular_l2=0.0
lora_dropout=0.0
learning_rate=5e-06
use_generative_next_token_loss="true"
max_len_train=512
max_len_val=512
current_dir=$(pwd)

source  $current_dir/experiments/lora_ensembles/hf_env.sh
for member_id in $(seq 0 $members); do
   ./srun_single.sh ./run.sh  $current_dir/experiments/lora_ensembles/run_lora_ens_train.py --batch_size $batch_size --epochs $epochs --member_id $member_id --train_dataset $train_dataset --learning_rate $learning_rate --lora_l2 $lora_l2 --regular_l2 $regular_l2 --lora_dropout $lora_dropout --use_generative_next_token_loss $use_generative_next_token_loss --max_len_train $max_len_train --max_len_val $max_len_val
done
