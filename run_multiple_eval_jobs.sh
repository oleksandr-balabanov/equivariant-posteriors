#!/bin/bash
# Define the number of epochs and members
epochs=1  # Example: 10 epochs
members=0  # Example: 5 members

for epoch in $(seq 1 $epochs); do
    for member_id in $(seq 0 $members); do
        sbatch run_eval_job.sh $epoch $member_id
    done
done