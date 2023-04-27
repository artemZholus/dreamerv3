#!/bin/bash
set -e
export IMAGE_PATH=$SCRATCH/jax_cuda11.8_latest.sif
export CODE_PATH=$SLURM_TMPDIR/drv3
mkdir -p $CODE_PATH
mkdir -p $(pwd)/wandb
echo "copying code into: $CODE_PATH"
export RUNNABLE="$@"
echo "Running: $RUNNABLE"
cp -r .netrc $CODE_PATH/
singularity exec --nv \
    -H $CODE_PATH:/home \
    -B $(pwd)/wandb:/home/wandb \
    $IMAGE_PATH $RUNNABLE