# Uncertainty quantification in fine-tuned LLMs using LoRA ensembles

This repository contains the code associated with the paper:

> Uncertainty quantification in fine-tuned LLMs using LoRA ensembles
> 
> Oleksandr Balabanov, Hampus Linander
> 
> [arXiv:2402.12264](https://arxiv.org/abs/2402.12264)

Although the repo is a fork of [hlinander/equivariant-posteriors](https://github.com/hlinander/equivariant-posteriors), it serves as an independent research project that reuses some training infrastructure. The goal is to fine-tune ensembles of LoRA LLM models on multiple-choice question datasets and evaluate their predictive uncertainties.

---

## Repository Overview

```
.
├── experiments
│ └── lora_ensembles
│ ├── datasets/ # Dataset configs for training & evaluation
│ ├── eval/ # Evaluation utilities for multiple-choice QA
│ ├── pretrained_models/ # Paths/pointers to HF checkpoints for Llama2 & Mistral
│ ├── train/ # Training configuration & scheduling
│ ├── utils/ # Auxiliary utilities (file ops, metrics, etc.)
│ ├── hf_env.sh # Environment variables for Hugging Face offline & cache settings 
│ ├── pipeline_lora_ens_eval.py # End-to-end evaluation pipeline of one LoRA ensemble member 
│ └── pipeline_lora_ens_train.py # End-to-end training pipeline of one LoRA ensemble member 
├── lib
│ ├── datasets/ # Dataset definitions & transformations
│ ├── data_factory.py # Functions for creating dataset objects
│ ├── data_registry.py # Registration logic for known datasets
│ ├── models/ # Model definitions & LoRA modules
│ ├── model_factory.py # Factory for LLM model creation
│ ├── train.py # Core training loop
│ ├── train_dataclasses.py # Config dataclasses for training
│ └── train_distributed.py # Distributed training support
├── run_multi_eval_jobs.sh # Script to launch parallel ensemble eval jobs
├── run_multi_train_jobs.sh # Script to launch parallel ensemble train jobs
└── ...
```
---

## Installation
1. Clone this repository: git clone https://github.com/oleksandr-balabanov/equivariant-posteriors.git
2. Set up the required environment variables and paths for your computing cluster.
We use SLURM to run jobs, and the following environment variables must be set:
```
export ENTVAR=                   # Base directory for your project files, models, and container image
export SLURM_PROJECT=            # SLURM project account name (used for job accounting)
export SLURM_PARTITION=          # SLURM partition to submit jobs to (e.g., "alvis")
```
3. Create a Singularity image and save it to $ENTVAR/equivariant-posteriors/image.img. This project uses [Nix](https://nixos.org/) to manage dependencies and build the Singularity image. Refer to the flake.nix file for configuration details. Alternatively, one may choose to build the Singularity image using a different method. In addition to other dependencies, the following deep learning packages are required:
```
torch==2.1.2
datasets==2.15.0
accelerate==0.24.1
transformers==4.35.2
peft==0.6.2
```
 
4. Select parameters and run:
```
run_multi_train_jobs.sh
run_multi_eval_jobs.sh
```
> Note: If needed, a wider variety of parameters can be adjusted in the train and eval config files:
```
/experiments/lora_ensembles/train/lora_ens_train_config_dataclass.py
/experiments/lora_ensembles/eval/lora_ens_member_eval_config_dataclass.py
```
> By default, the values defined in these config files are used unless they are overridden by parameters in the run scripts.

> Important: The evaluation config includes the training ensemble member configuration, which must be identical to that of the models intended to be loaded—except for the epoch number, which is set dynamically in the code. This config is selected by the runner, which locates the corresponding checkpoints based on the specified parameters. The loaded checkpoints are then used to perform evaluation on the specified dataset.
---

## Usage

### 1. Dataset Preparation
- Currently supports multiple-choice QA datasets MMLU and CommonsenseQA via Hugging Face.
- The dataset creation configs are found under experiments/lora_ensembles/datasets/.
- Datasets are registered in lib/data_registry.py and instantiated via lib/data_factory.py.

### 2. Model Configuration
- Supported model families: Llama2 and Mistral.
- To create a model, you need a configuration specifying:
  - The base model checkpoint (e.g. llama2-7b or mistral-7b)
  - LoRA fine-tuning hyperparameters (rank, alpha, dropout, etc.)
- Models are registered in lib/models and constructed through lib/model_factory.py.

### 3. Training
- The end-to-end training pipeline for a single ensemble member is located in experiments/lora_ensembles/pipeline_lora_ens_train.py.
- It uses a TrainRun config for each ensemble member, specifying data, model, and training hyperparameters.
- By default, checkpoints are saved at the end of each epoch. Training automatically resumes from the latest checkpoint, using the training configuration that is serialized and saved alongside the checkpoints.
- You can launch ensemble training in parallel using: experiments/lora_ensembles/run_multi_train_jobs.sh 

### 4. Evaluation
- Evaluation for multiple-choice QA tasks is handled by experiments/lora_ensembles/pipeline_lora_ens_eval.py.
- This script loads the fine-tuned ensemble member checkpoints and computes:
  - Softmax outputs for each possible answer token.
  - (Prefered) Reduced softmax over only the relevant answer tokens (eval_tokens) to minimize memory usage.
- Two evaluation modes are supported:
  - single_token: Compute loss/accuracy only over the single token representing the chosen answer.
  - next_token: Consider all tokens (question + answer choices + formatting).
- Use experiments/lora_ensembles/run_multi_eval_jobs.sh to evaluate multiple ensemble members in parallel.

> Note: The evaluation step only produces softmax probability files. Final metrics and uncertainty analyses (as presented in the paper) require additional post-processing, which is performed externally (e.g., in a separate notebook).
---

## Citation
If you find this work useful, please cite our paper:

@misc{balabanov2024uncertaintyquantificationfinetunedllms,
      title={Uncertainty quantification in fine-tuned LLMs using LoRA ensembles}, 
      author={Oleksandr Balabanov and Hampus Linander},
      year={2024},
      eprint={2402.12264},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.12264}, 
}
