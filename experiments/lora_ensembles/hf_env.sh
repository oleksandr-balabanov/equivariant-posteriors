export ENTVAR= #Set a base directory for storing model files and cache.  
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=$ENTVAR/LLM/huggingface_models
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_CACHE=$ENTVAR/LLM/huggingface_models