#!/bin/bash
# Define the number of epochs and members
epochs=0 # Example: 10 epochs
members=0 # Example: 5 members
eval_dataset="commonsense_qa"
train_dataset="commonsense_qa"
current_dir=$(pwd)

source  $current_dir/experiments/lora_ensembles/hf_env.sh
for epoch in $(seq 0 $epochs); do
    for member_id in $(seq 0 $members); do
        ./srun_single.sh ./run.sh  $current_dir/experiments/lora_ensembles/run_lora_ens_eval.py --epoch $epoch --member_id $member_id --eval_dataset $eval_dataset --train_dataset $train_dataset
    done
done

