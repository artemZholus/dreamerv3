#!/bin/bash
#SBATCH --partition=long
#SBATCH --time=1-00:00
#SBATCH -c 24
#SBATCH --mem 128G 
#SBATCH --gres=gpu:rtx8000:1
module load singularity/3.7.1
# on mila
# put your sif image here (better on a salloc'ed node):
# export SINGULARITY_CACHEDIR=$SLURM_TMPDIR/.singularity
# module load singularity 
# cd $SCRATCH
# singularity pull docker://artemzholus/jax_cuda11.8
export WANDB_MODE=online
export WANDB_ENTITY=chandar-rl
export WANDB_PROJECT=s4dreamer
./job.sh "$@"
