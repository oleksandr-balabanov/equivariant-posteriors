#!/bin/sh
set -x
singularity shell --nv --no-home --cleanenv --env PYTHONNOUSERSITE=1 $ENTVAR/equivariant-posteriors/image.img bash
