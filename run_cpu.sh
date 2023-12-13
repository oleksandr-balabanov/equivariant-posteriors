#!/bin/sh
export PYTHONPATH=$(pwd)
export TORCH_DEVICE="cpu"
python $@
