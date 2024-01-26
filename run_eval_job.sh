#!/bin/env bash

#SBATCH -A NAISS2023-5-353 -p alvis 
#SBATCH -t 0-24:00:00
#SBATCH -J eval_job_$1_member_$2
#SBATCH --gpus-per-node=A40:1
#SBATCH --output=./slurm_logs/myjob_%j.log
#SBATCH --error=./slurm_logs/myjob_%j.log

epoch=$1
member_id=$2

current_dir=$(pwd)
echo "The current working directory is: $current_dir"

sh  $current_dir/bash_singularity.sh
source  $current_dir/experiments/lora_ensembles/hf_env.sh
sh run.sh  $current_dir/experiments/lora_ensembles/test.py