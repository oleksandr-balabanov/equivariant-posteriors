#!/bin/bash
# Define the number of epochs and members
epochs=10 # Example: 10 epochs
members=4  # Example: 5 members
current_dir=$(pwd)

source  $current_dir/experiments/lora_ensembles/hf_env.sh
for epoch in $(seq 0 $epochs); do
    for member_id in $(seq 0 $members); do
        ./srun_single.sh ./run.sh  $current_dir/experiments/lora_ensembles/run_lora_ens_eval.py --epoch $epoch --member_id $member_id
    done
done

