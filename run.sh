#!/bin/sh
set -x
export PYTHONPATH=$(pwd)
export PYTHONBREAKPOINT=ipdb.set_trace
ulimit -n 64000
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 $@
