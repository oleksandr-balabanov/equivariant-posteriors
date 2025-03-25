# Uncertainty Quantification in Fine-Tuned LLMs Using LoRA Ensembles

This repository contains the code associated with the paper:

> Uncertainty Quantification in Fine-Tuned LLMs Using LoRA Ensembles
> Oleksandr Balabanov, Hampus Linander
> [arXiv:2402.12264](https://arxiv.org/abs/2402.12264)

Although it is a fork of [hlinander/equivariant-posteriors](https://github.com/hlinander/equivariant-posteriors), it has evolved into a largely independent project. The goal is to train ensembles of LoRA fine-tuned models on multiple-choice question datasets and evaluate their predictive uncertainties.

---

## Features
- Multiple-choice QA support (currently MMLU and CommonsenseQA only).
- LoRA-based fine-tuning of large language models (LLMs) such as Llama 2 and Mistral.
- Ensemble training: Easily configure and train multiple ensemble members.
- Softmax probability outputs for each ensemble member, which can be aggregated externally for further performance and uncertainty analyses.
- Modular design for datasets, models, training, and evaluation.

---

## Repository Overview

.
├── experiments
│ └── lora_ensembles
│ ├── datasets/ # Dataset configs for training & evaluation
│ ├── eval/ # Evaluation utilities for multiple-choice QA
│ ├── pretrained_models/ # Paths/pointers to HF checkpoints for Llama2 & Mistral
│ ├── train/ # Training configuration & scheduling
│ ├── utils/ # Auxiliary utilities (file ops, metrics, etc.)
│ ├── pipeline_lora_ens_eval.py # End-to-end evaluation pipeline
│ ├── pipeline_lora_ens_train.py # End-to-end training pipeline
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

---

## Installation
1. Clone this repository:
 bash  git clone https://github.com/<your-username>/<this-repo>.git  cd <this-repo> 
2. Install dependencies (preferably in a virtual environment):
 bash  pip install -r requirements.txt 
3. (Optional) Set up any environment variables or paths for your computing cluster as needed.

---

## Usage

### 1. Dataset Preparation
- Currently supports multiple-choice QA datasets MMLU and CommonsenseQA via Hugging Face.
- The dataset creation configs are found under experiments/lora_ensembles/datasets/.
- Datasets are registered in lib/data_registry.py and instantiated via lib/data_factory.py.

### 2. Model Configuration
- Supported model families: Llama 2 and Mistral.
- To create a model, you need a configuration specifying:
 - The base model checkpoint (e.g. llama2-7b or mistral-7b)
 - LoRA fine-tuning hyperparameters (rank, alpha, dropout, etc.)
- Models are registered in lib/models and constructed through lib/model_factory.py.

### 3. Training
- The end-to-end ensemble training pipeline is in experiments/lora_ensembles/pipeline_lora_ens_train.py.
- It uses a TrainRun config for each ensemble member, specifying data, model, and training hyperparameters.
- By default, checkpoints are saved at each epoch. Training is resumed automatically using the latest checkpoint.
- You can launch ensemble training in parallel using: experiments/lora_ensembles/run_multi_train_jobs.sh 

### 4. Evaluation
- Evaluation for multiple-choice QA tasks is handled by experiments/lora_ensembles/pipeline_lora_ens_eval.py.
- This script loads the fine-tuned ensemble member checkpoints and computes:
 - Softmax outputs for each possible answer token.
 - (Optional/Prefered) Reduced softmax over only the relevant answer tokens (eval_tokens) to minimize memory usage.
- Two evaluation modes are supported:
 - single_token: Compute loss/accuracy only over the single token representing the chosen answer.
 - next_token: Consider all tokens (question + answer choices + formatting).
- Use experiments/lora_ensembles/run_multi_eval_jobs.sh to evaluate multiple ensemble members in parallel.

> Note: The evaluation step only produces softmax probability files. Final metrics and uncertainty analyses (as presented in the paper) require additional post-processing, which is performed externally (e.g., in a separate notebook).

---

## License
This project inherits the LICENSE from the original fork. See LICENSE for details.

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

For inquiries feel free to open an issue or contact the authors directly.
