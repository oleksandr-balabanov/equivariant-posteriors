#!/bin/env bash
#SBATCH -A NAISS2023-22-544 -p alvis # find your project with the "projinfo" command
#SBATCH -t 0-24:00:00
#SBATCH -J EL
#SBATCH --gpus-per-node=T4:1


module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
module load IPython/7.25.0-GCCcore-10.3.0
module load matplotlib/3.4.2-foss-2021a
module load TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1
module load jax/0.3.9-foss-2021a-CUDA-11.3.1
module load AlphaFold/2.2.2-foss-2021a-CUDA-11.3.1
module load TensorFlow-Datasets/4.7.0-foss-2021a-CUDA-11.3.1
module load torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1

# install
#pip install tensorflow_datasets


# You can launch jupyter notebook or lab, but you must specify the config file as below: 
srun jupyter notebook