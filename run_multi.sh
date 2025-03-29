#!/bin/sh
set -x
export PYTHONPATH=$(pwd)
export PYTHONBREAKPOINT=ipdb.set_trace
ulimit -n 64000
python $@
